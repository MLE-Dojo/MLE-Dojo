"""
Kaggle Environment Module for Machine Learning Competitions

This module provides a Gymnasium-compatible environment for interacting with
Kaggle-style machine learning competitions. It integrates competition management,
code execution, feedback generation, and performance tracking in a safe and
extensible framework.

The environment supports various actions including information requests, code
validation, code execution, and history tracking, with comprehensive logging
and state management.
"""

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json
import logging
from datetime import datetime
from mledojo.gym.competition import CompetitionRegistry
from mledojo.gym.interface import Interface
from mledojo.gym.sandbox import Sandbox
from mledojo.gym.feedback import FeedbackManager
import math


class KaggleEnvironment(gym.Env):
    """
    A Gymnasium environment for Kaggle-style machine learning competitions.
    
    This environment provides a standardized interface for interacting with ML competitions,
    including code validation, execution, feedback generation, and performance tracking.
    It integrates sandbox execution for safety, comprehensive logging, and flexible
    action handling through a modular Interface system.
    
    The environment follows the Gymnasium API while providing additional features
    specific to ML competition workflows.
    """

    @classmethod
    def make(cls, competition_name: str, output_dir: str,
             competition_registry: Optional[CompetitionRegistry] = None,
             render_mode: Optional[str] = None,
             gpu_device: Optional[int] = None,
             gpu_memory_limit: Optional[int] = None,
             cpu_time_limit: Optional[int] = None,
             memory_limit: Optional[int] = None,
             execution_timeout: Optional[int] = None,
             custom_interface: Optional[Interface] = None,
             custom_feedback_manager: Optional[FeedbackManager] = None,
             score_mode: str = "position") -> "KaggleEnvironment":
        """
        Factory method to create a KaggleEnvironment instance with optional custom components.

        Args:
            competition_name: The name of the competition to load
            output_dir: Directory where output files (history, logs) are stored
            competition_registry: Optional registry containing competition definitions
            render_mode: Rendering mode ('human' or None)
            gpu_device: GPU device ID for sandbox execution
            gpu_memory_limit: GPU memory limit in GB
            cpu_time_limit: CPU time limit in seconds
            memory_limit: Memory limit in GB
            execution_timeout: Timeout for code execution in seconds
            custom_interface: Optional custom Interface instance
            custom_feedback_manager: Optional custom FeedbackManager instance
            score_mode: Scoring mode ('position' or 'raw')

        Returns:
            A fully configured KaggleEnvironment instance
        """
        env = cls(
            competition_name=competition_name,
            competition_registry=competition_registry,
            output_dir=output_dir,
            render_mode=render_mode,
            gpu_device=gpu_device,
            gpu_memory_limit=gpu_memory_limit,
            cpu_time_limit=cpu_time_limit,
            memory_limit=memory_limit,
            execution_timeout=execution_timeout,
            interface=custom_interface,
            feedback_manager=custom_feedback_manager,
            score_mode=score_mode
        )
        return env

    def __init__(self, competition_name: str, output_dir: str,
                 competition_registry: Optional[CompetitionRegistry] = None,
                 render_mode: Optional[str] = None,
                 gpu_device: Optional[int] = None,
                 gpu_memory_limit: Optional[int] = None,
                 cpu_time_limit: Optional[int] = None,
                 memory_limit: Optional[int] = None,
                 execution_timeout: Optional[int] = None,
                 interface: Optional[Interface] = None,
                 feedback_manager: Optional[FeedbackManager] = None,
                 feedback_mode: Optional[List[str]] = None,
                 score_mode: str = "position"):
        """
        Initialize the KaggleEnvironment with competition and modular components.

        Args:
            competition_name: Name of the competition to initialize
            output_dir: Directory for storing history and logs
            competition_registry: Registry containing competition definitions
            render_mode: Optional rendering mode ('human' or None)
            gpu_device: GPU device ID for sandboxed execution
            gpu_memory_limit: GPU memory limit in GB
            cpu_time_limit: CPU time limit in seconds
            memory_limit: Memory limit in GB
            execution_timeout: Execution timeout in seconds
            interface: Optional custom Interface instance
            feedback_manager: Optional custom FeedbackManager instance
            feedback_mode: Mode for feedback generation (default is ["base"])
            score_mode: Scoring mode ('position' or 'raw')
        """
        super().__init__()

        # Setup file paths for history and logging
        self.output_dir = Path(output_dir)
        self.render_mode = render_mode
        self.history_dir = self.output_dir / "env_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "history.json"
        self.log_file = self.history_dir / "env.log"

        # Initialize core components with dependency injection for flexibility
        self.competition_registry = competition_registry
        assert self.competition_registry.get(competition_name) is not None, "Competition not found in registry"
        self.feedback_mode = feedback_mode or ["base"]
        self.score_mode = score_mode
        assert self.score_mode in ["position", "raw"], "Score mode must be 'position' or 'raw'"
        self.sandbox = Sandbox(
            gpu_device=gpu_device,
            gpu_memory_limit=gpu_memory_limit,
            cpu_time_limit=cpu_time_limit,
            memory_limit=memory_limit,
            execution_timeout=execution_timeout,
            log_dir=self.history_dir / "sandbox.log"
        )

        # Load the competition
        self.competition_name = competition_name
        self.competition = self.competition_registry.get(competition_name)
        self.data_dir = self.competition.get_data_path() / "public"
        self.eval_dir = self.competition.get_data_path() / "private"

        # Initialize interface and feedback manager
        self.interface = interface if interface is not None else Interface(self.competition, self.output_dir)
        self.feedback_manager = feedback_manager if feedback_manager is not None else FeedbackManager()

        # Define action space: mapped to interface operations and environment controls
        self.actions = ["request_info", "validate_code", "execute_code", "reset", "get_history"]
        self.action_space = spaces.Discrete(len(self.actions))

        # Define observation space: extensible dictionary for rich state information
        self.observation_space = spaces.Dict({
            "feedback": spaces.Text(max_length=50000),  # Latest feedback message
            "total_reward": spaces.Box(low=-float('inf'), high=float('inf'), shape=()),  # Cumulative reward
            "current_raw_score": spaces.Box(low=-float('inf'), high=float('inf'), shape=()),  # Latest raw score
            "current_position_score": spaces.Box(low=-float('inf'), high=float('inf'), shape=()),  # Latest position score
            "history_summary": spaces.Text(max_length=50000)  # Summary of action history
        })

        # Initialize internal state
        self.total_reward = 0.0
        self.current_raw_score = 0.0
        self.current_position_score = 0.0
        self.best_code = None
        self.best_raw_score = None
        self.best_position_score = None
        self.step_count = 0
        self.logger = self._setup_logger()
        self._init_history()

    def step(self, action: str, **kwargs) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute an action and return the updated state of the environment.

        This method follows the Gymnasium API, executing the specified action,
        updating the environment state, and returning the observation, reward,
        termination status, truncation status, and additional info.

        Args:
            action: String name of the action to execute (must be in self.actions)
            **kwargs: Additional arguments specific to the action (e.g., 'code' for execute_code)

        Returns:
            Tuple containing:
                - observation: Dict with feedback and state information
                - reward: Float reward for the action
                - terminated: Boolean indicating if the episode has ended
                - truncated: Boolean indicating if truncated
                - info: Additional metadata about the action execution

        Raises:
            ValueError: If the action is invalid or execution fails
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.actions}")
            
        self.step_count += 1
        self.logger.info(f"Step {self.step_count}: Executing action '{action}'")
        if action == "request_info":
            self.logger.info(f"Action kwargs: {kwargs.get('info_type')}")

        try:
            # Execute action and get result
            result, reward = self._execute_action(action, **kwargs)
        except Exception as e:
            raise ValueError(f"Error in step for action '{action}': {str(e)}")
        
        try:
            # Generate feedback if applicable
            feedback = self._generate_feedback(action, result) 
        except Exception as e:
            raise ValueError(f"Error in feedback generation for action '{action}': {str(e)}")
        
        self.logger.info(f"Result status: {result.get('status')}")
        if action in ["validate_code", "execute_code"]:
            self.logger.info(f"Feedback: {feedback}")
        
        # Update state and history
        self.total_reward += reward
        self._update_history(action_name=action, result=result, feedback=feedback)

        # Construct and return observation
        observation = self._build_observation(result, feedback)
        info = {"action": action, "status": result.get("status", "UNKNOWN")}
        
        # In this environment, episodes don't terminate naturally
        terminated = False
        truncated = False
        
        return observation, reward

    def _execute_action(self, action: str, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Execute the appropriate action handler method.
        
        Args:
            action: The action name to execute
            **kwargs: Additional arguments for the action
            
        Returns:
            Tuple of (result_dict, reward)
            
        Raises:
            ValueError: If no handler is implemented for the action
        """
        action_handler = getattr(self, f"_handle_{action}", None)
        if action_handler:
            return action_handler(**kwargs)
        else:
            raise ValueError(f"No handler implemented for action: {action}")
    
    def _generate_feedback(self, action: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback with enhanced context.
        
        Args:
            action: The action that was executed
            result: The result dictionary from the action
            
        Returns:
            Dictionary containing feedback for each requested feedback mode
        """
        feedback_request = {}
        env_context = {
            "total_reward": self.total_reward,
            "current_raw_score": self.current_raw_score,
            "current_position_score": self.current_position_score,
            "best_raw_score": self.best_raw_score,
            "best_position_score": self.best_position_score,
            "step_count": self.step_count,
            "competition_name": self.competition_name,
            "history": self._get_history_summary(),
            "score_mode": self.score_mode
        }
        
        for mode in self.feedback_mode:
            feedback_request[mode] = {
                "interface_mode": action, 
                "raw_results": result,
                "env_context": env_context
            }
        
        return self.feedback_manager.get_feedback(feedback_request)
    
    def _build_observation(self, result: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the observation dictionary from current state.
        
        Args:
            result: The result dictionary from the action
            feedback: The feedback dictionary from the feedback manager
            
        Returns:
            Dictionary containing the current observation state
        """
        return {
            "action_status": result.get("status"),
            "feedback": feedback,
            "current_raw_score": self.current_raw_score,
            "current_position_score": self.current_position_score,
            "best_raw_score": self.best_raw_score,
            "best_position_score": self.best_position_score,
            "history_summary": self._get_history_summary()
        }

    def _handle_request_info(self, info_type: str = "all", **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Handle the request_info action.
        
        Args:
            info_type: Type of information to request (default: "all")
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result_dict, reward)
        """
        result = self.interface.info.get_info(info_type)
        reward = 0.0
        return result, reward

    def _handle_validate_code(self, code: str = "", **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Handle the validate_code action.
        
        Args:
            code: The code to validate
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result_dict, reward)
            
        Raises:
            ValueError: If code parameter is empty
        """
        if not code:
            raise ValueError("Code parameter is required for 'validate_code' action")
        result = self.interface.code_validation.validate(code, self.sandbox, self.output_dir)
        reward = 0.0
        return result, reward

    def _handle_execute_code(self, code: str = "", **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Handle the execute_code action.
        
        Args:
            code: The code to execute
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result_dict, reward)
            
        Raises:
            ValueError: If code parameter is empty
        """
        if not code:
            raise ValueError("Code parameter is required for 'execute_code' action")
        result = self.interface.code_execution.execute(code, self.sandbox, self.competition, self.output_dir, self.score_mode)
        reward = 0.0
        execution_result = result.get("submission", {})
        if not execution_result:
            return result, 0.0
        if execution_result.get("status") == "SUCCESS":
            raw_score = execution_result.get("raw_score", 0.0)
            position_score = execution_result.get("position_score", {})
            avg_position_score = position_score.get("avg_score", 0.0)
            
            # Update current scores
            self.current_raw_score = raw_score
            self.current_position_score = avg_position_score
            
            # Use appropriate score as reward based on score_mode
            if self.score_mode == "position":
                reward = avg_position_score
            else:  # raw mode
                reward = raw_score
            
            # Update best code and scores if applicable
            if self._is_better_score(raw_score, self.best_raw_score):
                self.best_raw_score = raw_score
                self.best_position_score = avg_position_score
                self.best_code = code
                
        return result, reward

    def _handle_reset(self, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Handle the reset action.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (observation_dict, reward)
        """
        observation, _ = self.reset()
        return observation, 0.0

    def _handle_get_history(self, **kwargs) -> Tuple[Dict[str, Any], float]:
        """
        Handle the get_history action.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (history_dict, reward)
        """
        result = self._get_history()
        return result, 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Optional seed for reproducibility
            options: Optional dictionary for additional reset options

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        self.total_reward = 0.0
        self.current_raw_score = 0.0
        self.current_position_score = 0.0
        self.best_code = None
        self.best_raw_score = None
        self.best_position_score = None
        self.step_count = 0
        self._init_history()
        self.logger.info("Environment reset")

        observation = {
            "feedback": "Environment reset successfully",
            "total_reward": 0.0,
            "current_raw_score": 0.0,
            "current_position_score": 0.0,
            "history_summary": "History initialized"
        }
        return observation, {}

    def render(self):
        """
        Render the current state of the environment based on render_mode.
        
        For 'human' mode, outputs the current state to the logger.
        """
        if self.render_mode == "human":
            history = self._get_history()["data"]
            self.logger.info(
                f"Step: {self.step_count}, Total Reward: {self.total_reward}, "
                f"Current Raw Score: {self.current_raw_score}, Current Position Score: {self.current_position_score}, "
                f"Actions Taken: {len(history['actions'])}"
            )

    def close(self):
        """
        Clean up resources, such as closing log handlers.
        """
        self.logger.info("Closing environment")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # Helper Methods for History and Logging

    def _setup_logger(self) -> logging.Logger:
        """
        Configure and return a logger for the environment.

        Returns:
            A configured logging.Logger instance writing to a file.
        """
        logger = logging.getLogger(f"kaggle_env_{self.competition_name}")
        logger.setLevel(logging.INFO)
        # Prevent messages from propagating to the root logger's handlers
        logger.propagate = False
        logger.handlers = []  # Clear any existing handlers if re-configuring
        
        # File handler for persistent logging (env.log)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        return logger

    def _init_history(self):
        """
        Initialize or reset the history file with default values.
        """
        history = {
            "competition": self.competition_name,
            "actions": [],
            "total_reward": 0.0,
            "best_code": None,
            "best_raw_score": None,
            "best_position_score": None,
            "step_count": 0,
            "created_at": datetime.now().isoformat()
        }
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _update_history(self, action_name: str, result: Dict[str, Any], feedback: Union[str, Dict[str, Any]]):
        """
        Append an action record to the history file and update state.

        Args:
            action_name: Name of the action executed
            result: Result dictionary from the action
            feedback: Feedback from the FeedbackManager
        """
        with open(self.history_file, 'r') as f:
            history = json.load(f)

        history["actions"].append({
            "step": self.step_count,
            "action": action_name,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "feedback": feedback
        })
        history["total_reward"] = self.total_reward
        history["step_count"] = self.step_count
        history["best_code"] = self.best_code
        history["best_raw_score"] = self.best_raw_score
        history["best_position_score"] = self.best_position_score

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _get_history(self) -> Dict[str, Any]:
        """
        Retrieve the full history from the history file.

        Returns:
            Dictionary with status and history data
        """
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        return {"status": "SUCCESS", "data": history}

    def _get_history_summary(self) -> str:
        """
        Generate a concise summary of the action history.

        Returns:
            A string summarizing the history
        """
        history = self._get_history()["data"]
        actions = history["actions"]
        if not actions:
            return "No actions taken yet."
        summary = f"Total Actions: {len(actions)}, Last Action: {actions[-1]['action']}"
        return summary

    def _is_better_score(self, new_score: float, best_score: Optional[float]) -> bool:
        """
        Determine if the new score is better than the best score based on competition metrics,
        correctly handling NaN values.

        Args:
            new_score: The new score to evaluate
            best_score: The current best score (None if no best score yet)

        Returns:
            Boolean indicating if new_score is better
        """
        # If the new score is NaN, it can never be better.
        if new_score is None or math.isnan(new_score):
            return False
        
        # If there is no best score yet, or the current best is NaN, the new valid score is better.
        if best_score is None or math.isnan(best_score):
            return True
            
        # Otherwise, compare based on the competition's metric direction.
        higher_is_better = self.competition.create_metrics().higher_is_better
        return (new_score > best_score) if higher_is_better else (new_score < best_score)
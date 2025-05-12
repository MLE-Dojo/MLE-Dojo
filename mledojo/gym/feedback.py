"""
Feedback Module for Machine Learning Environment

This module provides a comprehensive feedback system for the MLE-Dojo environment.
It defines abstract interfaces and concrete implementations for different types of
feedback providers, including automated validation feedback, execution feedback,
LLM-based feedback, and human feedback.

The module is designed to be extensible, allowing custom feedback providers to be
registered and used within the environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional, Union


class Feedback(ABC):
    """
    Abstract base class defining the interface for all feedback providers.
    
    All feedback providers must implement the get_feedback method to generate
    feedback based on provided parameters.
    """
    
    @abstractmethod
    def get_feedback(self, **kwargs) -> Dict[str, Any]:
        """
        Generate feedback based on provided parameters.
        
        Args:
            **kwargs: Provider-specific parameters needed to generate feedback
        
        Returns:
            Dict with 'feedback_status' (SUCCESS/FAILED), 'feedback' (main content),
            and optional additional fields.
        """
        pass


class BaseFeedback(Feedback):
    """
    Generates automatic feedback based on validation and execution results.
    
    This class processes results from code validation, execution, and information
    requests to provide structured, human-readable feedback to the user.
    """
    
    def __init__(self):
        """Initialize with default feedback processors."""
        self.processors = {
            "validate_code": self._process_validation,
            "execute_code": self._process_execution, 
            "request_info": self._process_info
        }

    def register_processor(self, mode: str, processor: Callable[[Dict[str, Any], Dict[str, Any]], str]) -> None:
        """
        Register a new feedback processor for a specific mode.
        
        Args:
            mode: Mode to register the processor for (e.g., 'validate_code')
            processor: Function that takes raw results and environment context and returns a formatted feedback string
        """
        self.processors[mode] = processor

    def get_feedback(self, 
                    interface_mode: Optional[str] = None, 
                    raw_results: Optional[Dict[str, Any]] = None, 
                    env_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Summarize interface results in a concise, readable format.
        
        Args:
            interface_mode: Type of interface to process ('request_info', 'validate_code', or 'execute_code')
            raw_results: Raw results from the corresponding interface
            env_context: Additional environment context for enhanced feedback
            
        Returns:
            Dict with feedback_status, feedback string, and optional error information
        """
        if not interface_mode or interface_mode not in self.processors:
            return {
                "feedback_status": "FAILED",
                "error": f"Invalid interface mode: {interface_mode}, supported modes: {', '.join(self.processors.keys())}"
            }

        if not raw_results:
            raw_results = {}
            
        if not env_context:
            env_context = {}

        try:
            feedback_text = self.processors[interface_mode](raw_results, env_context)
            return {
                "feedback_status": "SUCCESS",
                "feedback": feedback_text
            }
        except Exception as e:
            return {
                "feedback_status": "FAILED", 
                "error": f"Error generating feedback: {str(e)}"
            }

    def _process_validation(self, validation_result: Dict[str, Any], env_context: Dict[str, Any]) -> str:
        """
        Process validation results and generate structured feedback.
        
        Args:
            validation_result: Results from code validation
            env_context: Additional environment context
            
        Returns:
            Formatted string with validation feedback
        """
        validation_template = """=== Code Validation Results ===
                            {syntax_feedback}
                            {runtime_feedback}
                            Code execution time: {execution_time}
                            """

        # Process syntax validation
        syntax = validation_result.get("syntax", {})
        if syntax.get("status") == "SUCCESS":
            syntax_feedback = "Syntax check passed: Valid Python code"
        else:
            syntax_feedback = self._format_error_feedback(syntax)

        # Process runtime validation  
        runtime = validation_result.get("runtime", {})
        if runtime.get("status") == "SUCCESS":
            runtime_feedback = "Runtime check passed: Code executes without errors"
            if "output" in runtime:
                runtime_feedback += f"\nCode output: {runtime.get('output')}"
        else:
            runtime_feedback = self._format_error_feedback(runtime)

        execution_time = runtime.get('execution_time', 'N/A')

        return validation_template.format(
            syntax_feedback=syntax_feedback,
            runtime_feedback=runtime_feedback,
            execution_time=execution_time
        )

    def _format_error_feedback(self, result: Dict[str, Any]) -> str:
        error_type = result.get('error_type', 'Unknown error')
        error_msg = result.get('error', 'Unknown error')
        details = result.get('details', 'No details available')
        causes = result.get('potential_causes', 'No potential causes available')
        return f"{error_type}: {error_msg}\n" \
               f"Error Details: {details}\n" \
               f"Potential Causes: {causes}"

    def _process_execution(self, execution_result: Dict[str, Any], env_context: Dict[str, Any]) -> str:
        """
        Process execution results and generate structured feedback.
        
        Args:
            execution_result: Results from code execution
            env_context: Additional environment context including best scores
            
        Returns:
            Formatted string with execution feedback
        """
        execution_template = """=== Code Execution Results ===
                            {execution_feedback}

                            {submission_feedback}
                            """

        # Process execution
        exec_result = execution_result.get("execution", {})
        if exec_result.get("status") == "SUCCESS":
            execution_feedback = f"Execution successful\nCode execution time: {exec_result.get('execution_time', 'N/A')}"
            if exec_result.get("output"):
                execution_feedback += f"\nCode output: {exec_result['output']}"
        else:
            execution_feedback = f"Execution failed: {exec_result.get('error', 'Unknown error')}"
            if "details" in exec_result:
                execution_feedback += f"\nError Details: {exec_result['details']}"
            if "potential_causes" in exec_result:
                execution_feedback += f"\nPotential Causes: {exec_result['potential_causes']}"

        # Process submission if available
        score_mode = env_context.get("score_mode", "position")
        submission = execution_result.get("submission", {})
        if submission:
            submission_template = """=== Submission Evaluation ===
                                {submission_status}
                                {leaderboard_info}
                                {score_info}
                                """
            
            if submission.get("status") == "SUCCESS":
                raw_score = submission.get('raw_score', 'N/A')
                submission_status = "Submission successful"
                
                # Add leaderboard positions
                position_info = submission.get("position_score", {})
                leaderboard_info = []
                avg_position_score = 0
                board_count = 0
                
                for board_type in ["private", "public"]:
                    if board_type in position_info:
                        board_pos = position_info[board_type]
                        position = board_pos.get('position')
                        total = board_pos.get('total')
                        leaderboard_info.append(
                            f"{board_type.capitalize()} Leaderboard: Position {position} / {total}"
                        )
                        if 'position_score' in board_pos:
                            avg_position_score += board_pos['position_score']
                            board_count += 1
                
                leaderboard_info_str = "\n".join(leaderboard_info) if score_mode == "position" else ""
                
                # Add scores
                score_info = [f"Raw Score: {raw_score}"]
                if board_count > 0:
                    avg_score = avg_position_score / board_count
                    score_info.append(f"Average Position Score: {avg_score:.4f}")
                score_info.append(f"Best Raw Score: {env_context.get('best_raw_score', 'N/A')}")
                score_info.append(f"Best Position Score: {env_context.get('best_position_score', 'N/A')}") if score_mode == "position" else ""
                score_info_str = "\n".join(score_info)
            else:
                submission_status = f"Submission error ({submission.get('error_type', 'unknown')}): {submission.get('error', 'Unknown')}"
                if "details" in submission:
                    submission_status += f"\nError details: {submission['details']}"
                leaderboard_info_str = ""
                score_info_str = ""
            
            submission_feedback = submission_template.format(
                submission_status=submission_status,
                leaderboard_info=leaderboard_info_str,
                score_info=score_info_str
            )
        else:
            submission_feedback = ""

        return execution_template.format(
            execution_feedback=execution_feedback,
            submission_feedback=submission_feedback
        ).strip()

    def _process_info(self, info_result: Dict[str, Any], env_context: Dict[str, Any]) -> str:
        """
        Process information request results and generate feedback.
        
        Args:
            info_result: Results from information request
            env_context: Additional environment context
            
        Returns:
            Formatted string with information feedback
        """
        info_template = """=== Competition Info ===
                        {info_content}
                        """
        
        if info_result and info_result.get("status") == "SUCCESS":
            info_content = f"Your requested information: {info_result.get('data', {})}"
        else:
            info_content = "No information available or request failed."
            
        return info_template.format(info_content=info_content)


class LLMFeedback(Feedback):
    """
    Provides feedback from a language model evaluating the code.
    
    This class can be integrated with external LLM APIs to provide
    AI-powered code analysis and suggestions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        """
        Initialize the LLM feedback provider.
        
        Args:
            api_key: Optional API key for LLM service
            model: Model identifier to use for feedback generation
        """
        self.api_key = api_key
        self.model = model
    
    def get_feedback(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Generate LLM feedback for the given code.
        
        Args:
            code: The source code string to evaluate
            **kwargs: Additional parameters for the LLM API
            
        Returns:
            Dict with feedback_status and LLM-generated feedback
        """
        # Placeholder for real LLM integration (e.g., API call)
        # In a real implementation, this would call an LLM API
        return {
            "feedback_status": "SUCCESS",
            "feedback": "LLM feedback not implemented yet. This is a placeholder."
        }


class HumanFeedback(Feedback):
    """
    Collects interactive feedback from a human user.
    
    This class provides mechanisms to request and collect feedback
    from human users, which can be useful for educational settings.
    """
    
    def get_feedback(self, prompt: str = "Please provide your feedback: ") -> Dict[str, Any]:
        """
        Prompt the user for feedback and return it.
        
        Args:
            prompt: Custom prompt string for user input
            
        Returns:
            Dict with feedback_status and user-provided feedback
        """
        try:
            # In a real implementation, this might use a UI component
            # or other interactive mechanism
            return {
                "feedback_status": "SUCCESS",
                "feedback": "Human feedback collection not implemented yet. This is a placeholder."
            }
        except Exception as e:
            return {
                "feedback_status": "FAILED",
                "error": f"Failed to collect human feedback: {str(e)}"
            }


class FeedbackManager:
    """
    Manages feedback providers and retrieves feedback based on user requests.
    
    This class serves as a central registry for feedback providers and handles
    the routing of feedback requests to the appropriate provider.
    """
    
    def __init__(self):
        """Initialize with default feedback providers."""
        self._providers: Dict[str, Feedback] = {
            "base": BaseFeedback(),
            "llm": LLMFeedback(),
            "human": HumanFeedback()
        }
    
    def register(self, feedback_type: str, provider: Feedback) -> None:
        """
        Register a new feedback provider dynamically.
        
        Args:
            feedback_type: Unique identifier for the feedback type
            provider: Instance of a Feedback subclass
        """
        if not isinstance(provider, Feedback):
            raise TypeError("Provider must be an instance of a Feedback subclass")
        self._providers[feedback_type] = provider
    
    def get_provider(self, feedback_type: str) -> Optional[Feedback]:
        """
        Get a specific feedback provider by type.
        
        Args:
            feedback_type: The type of feedback provider to retrieve
            
        Returns:
            The requested Feedback provider or None if not found
        """
        return self._providers.get(feedback_type)
    
    def get_feedback(self, requests: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve feedback for specified types with their parameters.
        
        Args:
            requests: Dict mapping feedback types to their keyword arguments.
                     e.g., {"base": {"interface_mode": "validate_code", "raw_results": {...}},
                            "human": {"prompt": "Your thoughts?"}}
        
        Returns:
            Dict mapping feedback types to their feedback results.
            Each result contains 'feedback_status', 'feedback', and optional error information.
        
        Example:
            >>> fm.get_feedback({"base": {"interface_mode": "validate_code", "raw_results": {...}}})
            {"base": {"feedback_status": "SUCCESS", "feedback": "..."}}
        """
        result = {}
        for feedback_type, kwargs in requests.items():
            if feedback_type not in self._providers:
                result[feedback_type] = {
                    "feedback_status": "FAILED",
                    "error": f"Unknown feedback type: {feedback_type}"
                }
                continue
            
            try:
                feedback = self._providers[feedback_type].get_feedback(**kwargs)
                result[feedback_type] = feedback
            except TypeError as e:
                result[feedback_type] = {
                    "feedback_status": "FAILED",
                    "error": f"Missing or invalid parameters: {str(e)}"
                }
            except Exception as e:
                result[feedback_type] = {
                    "feedback_status": "FAILED",
                    "error": f"Error generating feedback: {str(e)}"
                }
                
        return result

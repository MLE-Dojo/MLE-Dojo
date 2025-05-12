"""
MLE-Dojo Interface Module.

This module provides the main interfaces for interacting with the MLE-Dojo framework.
It includes components for information retrieval, code validation, and code execution
within a sandboxed environment.

The module is designed with a modular architecture that allows for dynamic registration
of new interface components, making it extensible for various competition needs.

Typical usage:
    interface = Interface(competition, output_dir)
    info = interface.info.get_info("overview")
    validation_result = interface.code_validation.validate(code, sandbox, output_dir)
    execution_result = interface.code_execution.execute(code, sandbox, competition, output_dir)
"""

import os
from typing import Dict, Any, Union, Callable, Optional, List
from pathlib import Path
import pandas as pd
import tempfile
import subprocess
import traceback
from datetime import datetime
from mledojo.gym.sandbox import Sandbox
from mledojo.gym.competition import Competition
from mledojo.gym.error import (
    MLEDojoError,
    SandboxError, 
    SubmissionNotFoundError,
    GroundTruthError,
    InvalidSubmissionError,
    EvaluationError,
    LeaderboardError,
    ArchiveError,
    InfoError,
    CodeSyntaxError,
    CodeRuntimeError,
    error_to_dict,
    create_error_response
)
from mledojo.gym.utils import (
    run_in_sandbox, 
    archive_file, 
    save_code_file,
    build_tree
)


# Main Interface Class
class Interface:
    """Main interface for competition interactions.
    
    This class provides a unified interface for interacting with competition components,
    allowing dynamic registration of new interface components.
    
    Attributes:
        components: Dictionary of registered interface components
    """
    
    def __init__(self, competition: Optional[Competition] = None, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the interface with competition and output directory.
        
        Args:
            competition: Competition instance to use
            output_dir: Directory for output files
        """
        # Dictionary to hold interface components
        self.components = {
            "request_info": InfoInterface(competition, output_dir),
            "validate_code": CodeValidationInterface(),
            "execute_code": CodeExecutionInterface()
        }
    
    def register(self, name: str, component: object) -> None:
        """Register a new interface component dynamically.
        
        Args:
            name: Name to register the component under
            component: Component instance to register
        """
        self.components[name] = component
    
    @property
    def info(self) -> "InfoInterface":
        """Access the info interface.
        
        Returns:
            InfoInterface instance
        """
        return self.components["request_info"]
    
    @property
    def code_validation(self) -> "CodeValidationInterface":
        """Access the code validation interface.
        
        Returns:
            CodeValidationInterface instance
        """
        return self.components["validate_code"]
    
    @property
    def code_execution(self) -> "CodeExecutionInterface":
        """Access the execution interface.
        
        Returns:
            CodeExecutionInterface instance
        """
        return self.components["execute_code"]
    

# Info Interface
class InfoInterface:
    """Handles retrieval of competition-related information.
    
    This class provides methods to access various types of competition information,
    such as overview, sample submissions, and data structure.
    
    Attributes:
        competition: Competition instance to retrieve information from
        output_dir: Directory for output files
        providers: Registry of information provider functions
    """
    
    def __init__(self, competition: Optional[Competition] = None, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the info interface with competition and output directory.
        
        Args:
            competition: Competition instance to use
            output_dir: Directory for output files
        """
        self.competition = competition
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Registry of info providers
        self.providers = {
            "overview": self._get_overview,
            "sample_submission": self._get_sample_submission, 
            "data_structure": self._get_data_structure,
            "data_path": lambda: str(self.competition.get_public_data_path()) if self.competition else None,
            "output_path": lambda: str(self.output_dir) if self.output_dir else None,
            "name": lambda: self.competition.name if self.competition else None,
            "metadata": lambda: vars(self.competition.info) if self.competition else None
        }

    def register_provider(self, info_type: str, provider: Callable[[], Any]) -> None:
        """Register a new info provider dynamically.
        
        Args:
            info_type: Type of information to register
            provider: Function that provides the information
        """
        self.providers[info_type] = provider

    def get_info(self, info_type: str = "all") -> Dict[str, Any]:
        """Retrieve competition information for a specific type or all types.
        
        Args:
            info_type: Type of information to retrieve or "all" for all types
            
        Returns:
            Dictionary containing the requested information
            
        Raises:
            InfoError: If information retrieval fails
            ValueError: If an invalid info type is requested
        """
        if not self.competition:
            raise InfoError(
                message="No competition configured",
                details="InfoInterface requires a competition instance to retrieve information"
            )
            
        if info_type != "all" and info_type not in self.providers:
            raise InfoError(
                message=f"Invalid info type: {info_type}",
                details=f"Available info types: {', '.join(self.providers.keys())}"
            )
        
        types_to_fetch = self.providers.keys() if info_type == "all" else [info_type]
        result = {}
        
        for key in types_to_fetch:
            try:
                result[key] = self.providers[key]()
            except Exception as e:
                raise InfoError(
                    message=f"Error retrieving {key}",
                    details=str(e)
                )
        
        return {"status": "SUCCESS", "data": result}

    def _get_overview(self) -> str:
        """Get competition overview from description file.
        
        Returns:
            Competition description text or default message if not available
            
        Raises:
            InfoError: If description file cannot be read
        """
        try:
            path = self.competition.get_data_path() / "public" / "description.txt"
            return path.read_text() if path.exists() else "No description available"
        except Exception as e:
            raise InfoError(
                message="Failed to read competition description",
                details=str(e)
            )

    def _get_sample_submission(self) -> Union[Dict, str]:
        """Get sample submission as a dictionary.
        
        Returns:
            Sample submission as dictionary or default message if not available
            
        Raises:
            InfoError: If sample submission file cannot be read
        """
        try:
            path = self.competition.get_data_path() / "public" / "sample_submission.csv"
            return pd.read_csv(path).head(3).to_dict() if path.exists() else "No sample submission available"
        except Exception as e:
            raise InfoError(
                message="Failed to read sample submission",
                details=str(e)
            )

    def _get_data_structure(self) -> str:
        """Get directory structure of competition data.
        
        Returns:
            Text representation of the competition data directory structure
            
        Raises:
            InfoError: If data structure cannot be generated
        """
        try:
            # Check if data structure file exists
            structure_file_path = self.competition.get_data_path() / "public" / "data_structure.txt"
            if structure_file_path.exists():
                return structure_file_path.read_text()
            
            # If file doesn't exist, generate the structure
            return build_tree(self.competition.get_data_path() / "public")
        except Exception as e:
            raise InfoError(
                message="Failed to generate data structure",
                details=str(e)
            )


# Code Validation Interface
class CodeValidationInterface:
    """Validates code syntax and runtime behavior.
    
    This class provides methods to check code syntax and execute code in a sandbox
    to validate its runtime behavior.
    """
    
    def validate(self, code: str, sandbox: Sandbox, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Validate code syntax and execution in a sandbox.
        
        Args:
            code: Code content to validate
            sandbox: Sandbox instance to use for validation
            output_dir: Optional directory to save validation files
            
        Returns:
            Dictionary containing validation results
            
        Raises:
            CodeSyntaxError: If code syntax is invalid
            CodeRuntimeError: If code execution fails
        """
        # Save validation code if output directory is provided
        code_path = None
        if output_dir:
            try:
                save_code_file(code, output_dir, "validation")
            except Exception as e:
                # Non-critical error, continue with validation
                pass
            
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                code_path = Path(temp_file.name)
            
            syntax_result = self._check_syntax(code_path)
            runtime_result = self._check_runtime(code_path, sandbox)
            
            result = {
                "syntax": syntax_result,
                "runtime": runtime_result,
                "status": "SUCCESS" if syntax_result["status"] == "SUCCESS" and runtime_result["status"] == "SUCCESS" else "FAILED"
            }
            
            # Remove any submission files created during validation
            if output_dir:
                submission_file = output_dir / "submission.csv"
                if submission_file.exists():
                    try:
                        submission_file.unlink()
                    except Exception:
                        # Non-critical error, continue
                        pass
                    
            return result
        finally:
            if code_path and code_path.exists():
                try:
                    code_path.unlink()
                except Exception:
                    # Non-critical error, continue
                    pass
    
    def _check_syntax(self, code_path: Path) -> Dict[str, Any]:
        """Check code syntax using py_compile.
        
        Args:
            code_path: Path to the code file to check
            
        Returns:
            Dictionary containing syntax check results
        """
        try:
            subprocess.check_output(
                ['python3', '-m', 'py_compile', str(code_path)],
                stderr=subprocess.STDOUT
            )
            return {"status": "SUCCESS", "details": "Code syntax is valid"}
        except subprocess.CalledProcessError as e:
            return create_error_response(CodeSyntaxError(
                message="Code syntax validation failed",
                details=e.output.decode()
            ))
    
    def _check_runtime(self, code_path: Path, sandbox: Sandbox) -> Dict[str, Any]:
        """Check code runtime behavior in a sandbox.
        
        Args:
            code_path: Path to the code file to check
            sandbox: Sandbox instance to use for execution
            
        Returns:
            Dictionary containing runtime check results
        """
        try:
            return run_in_sandbox(code_path, sandbox)
        except Exception as e:
            return create_error_response(CodeRuntimeError(
                message="Code runtime validation failed",
                details=str(e)
            ))


# Execution Interface
class CodeExecutionInterface:
    """Executes code and evaluates submissions.
    
    This class provides methods to execute code in a sandbox environment,
    process the resulting submission, and evaluate it against ground truth data.
    """
    
    def execute(self, code: str, sandbox: Sandbox, competition: Competition, output_dir: Path, score_mode: str = "position") -> Dict[str, Any]:
        """Execute code and process the resulting submission.
        
        Args:
            code: Code content to execute
            sandbox: Sandbox instance to use for execution
            competition: Competition instance for evaluation
            output_dir: Directory for output files
            score_mode: Mode for score calculation, either "position" or "raw"
            
        Returns:
            Dictionary containing execution and submission results
            
        Raises:
            SandboxError: If execution in sandbox fails
            SubmissionNotFoundError: If submission file is not found
            ArchiveError: If submission archiving fails
            MLEDojoError: For other unexpected errors
        """
        # Save execution code
        code_path = None
        try:
            save_code_file(code, output_dir, "execution")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                code_path = Path(temp_file.name)
            
            execution_result = run_in_sandbox(code_path, sandbox)
            if execution_result["status"] == "FAILED":
                return {
                    "status": "FAILED",
                    "execution": execution_result, 
                    "submission": None
                }
            
            submission_path = output_dir / competition.info.output_type
            if submission_path.exists():
                try:
                    archived_path = archive_file(submission_path, output_dir, "submission", competition.info.output_type.split('.')[-1])
                    submission_result = self._process_submission(competition, archived_path, output_dir, score_mode)
                except ArchiveError as e:
                    return {
                        "status": "FAILED",
                        "execution": execution_result,
                        "submission": create_error_response(e)
                    }
                except Exception as e:
                    return {
                        "status": "FAILED",
                        "execution": execution_result,
                        "submission": create_error_response(MLEDojoError(
                            message="Unexpected error during submission processing",
                            details=str(e)
                        ))
                    }
            else:
                submission_result = create_error_response(SubmissionNotFoundError(
                    message="Submission file not found",
                    details=f"No submission file found at {submission_path}"
                ))
            
            result = {
                "execution": execution_result,
                "submission": submission_result,
                "status": "SUCCESS" if execution_result["status"] == "SUCCESS" and 
                                    (submission_result is None or submission_result.get("status") == "SUCCESS") 
                                 else "FAILED"
            }
            return result
        except Exception as e:
            return {
                "status": "FAILED",
                "execution": create_error_response(MLEDojoError(
                    message="Unexpected error during code execution",
                    details=str(e)
                )),
                "submission": None
            }
        finally:
            if code_path and code_path.exists():
                try:
                    code_path.unlink()
                except Exception:
                    # Non-critical error, continue
                    pass
    
    def _process_submission(self, competition: Competition, submission_path: Path, output_dir: Path, score_mode: str = "position") -> Dict[str, Any]:
        """Process and evaluate a submission file.
        
        Args:
            competition: Competition instance for evaluation
            submission_path: Path to the submission file
            output_dir: Directory for output files
            score_mode: Mode for score calculation, either "position" or "raw"
            
        Returns:
            Dictionary containing submission processing results
            
        Raises:
            SubmissionNotFoundError: If submission file is not found
            InvalidSubmissionError: If submission format is invalid
            GroundTruthError: If ground truth data is missing or invalid
            EvaluationError: If submission evaluation fails
            LeaderboardError: If leaderboard processing fails
            MLEDojoError: For other unexpected errors
        """
        if not submission_path.exists():
            return create_error_response(SubmissionNotFoundError(
                message="Submission file not found",
                details=f"No submission file found at {submission_path}"
            ))
        
        try:
            metrics = competition.create_metrics()
            try:
                submission = pd.read_csv(submission_path)
            except Exception as e:
                return create_error_response(InvalidSubmissionError(
                    message="Failed to read submission file",
                    details=f"Could not parse submission file as CSV: {str(e)}"
                ))

            ground_truth_path = competition.get_data_path() / "private" / "test_answer.csv"
            if not ground_truth_path.exists():
                return create_error_response(GroundTruthError(
                    message="Ground truth file not found",
                    details=f"Expected ground truth file at {ground_truth_path}"
                ))

            try:
                ground_truth = pd.read_csv(ground_truth_path)
            except Exception as e:
                return create_error_response(GroundTruthError(
                    message="Failed to read ground truth file",
                    details=f"Could not parse ground truth file as CSV: {str(e)}"
                ))
            
            try:
                validation_msg = metrics.validate_submission(submission, ground_truth)
            except Exception as e:
                return create_error_response(InvalidSubmissionError(
                    message="Submission validation failed",
                    details=str(e)
                ))

            try:
                score = metrics.evaluate(ground_truth, submission)
            except Exception as e:
                return create_error_response(EvaluationError(
                    message="Failed to evaluate submission",
                    details=f"Error during metric calculation: {str(e)}"
                ))

            result = {
                "status": "SUCCESS",
                "raw_score": score,
                "details": "Submission processed successfully"
            }
            
            if score_mode == "position":
                try:
                    position = self._calculate_leaderboard_position(
                        score, competition.get_data_path() / "private", metrics.higher_is_better
                    )
                    result["position_score"] = position
                except Exception as e:
                    return create_error_response(LeaderboardError(
                        message="Failed to calculate leaderboard position",
                        details=f"Error processing leaderboard data: {str(e)}"
                    ))
            
            return result
            
        except Exception as e:
            return create_error_response(MLEDojoError(
                message="Unexpected error during submission processing",
                details=str(e)
            ))

    
    def _calculate_leaderboard_position(self, score: float, leaderboard_dir: Path, higher_is_better: bool) -> Dict[str, Any]:
        """Calculate leaderboard position and percentile.
        
        Args:
            score: Submission score to evaluate
            leaderboard_dir: Directory containing leaderboard files
            higher_is_better: Whether higher scores are better
            
        Returns:
            Dictionary containing position information
            
        Raises:
            LeaderboardError: If leaderboard processing fails
        """
        position_info = {}
        total_position_score = 0.0
        count = 0
        
        for board_type in ["private", "public"]:
            board_path = leaderboard_dir / f"{board_type}_leaderboard.csv"
            if not board_path.exists():
                continue
            
            try:
                leaderboard = pd.read_csv(board_path)
                print(leaderboard.head())
                scores = leaderboard.iloc[:, 1].values  # Assume scores in second column
                total = len(scores)
                
                sorted_scores = sorted(scores, reverse=higher_is_better)
                position = next(
                    (i + 1 for i, s in enumerate(sorted_scores) if (score >= s if higher_is_better else score <= s)),
                    total + 1
                )
                
                position_score = (total - position + 1) / total
                position_info[board_type] = {
                    "position": position,
                    "total": total,
                    "position_score": position_score,
                }
                
                total_position_score += position_score
                count += 1
            except Exception as e:
                raise LeaderboardError(
                    message=f"Failed to process {board_type} leaderboard",
                    details=str(e)
                )
        
        # Calculate average position score across private and public leaderboards
        if count > 0:
            position_info["avg_score"] = total_position_score / count
            
        return position_info

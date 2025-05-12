"""
MLE-Dojo Error Handling Module.

This module defines a comprehensive error hierarchy for the MLE-Dojo framework.
It provides specialized exception classes for different components of the system,
along with utility functions for error handling and reporting.

Typical usage:
    try:
        # Some operation that might fail
        raise SandboxExecutionError("Process terminated unexpectedly", "Exit code: 1")
    except MLEDojoError as e:
        error_dict = error_to_dict(e)
        # Handle or report the error
"""

from typing import Dict, Any, Optional, List, Union


class MLEDojoError(Exception):
    """Base class for all MLE-Dojo exceptions.
    
    This class serves as the foundation for the MLE-Dojo error hierarchy.
    It provides consistent error reporting with message, details, and
    potential causes.
    
    Attributes:
        message (str): Primary error message.
        details (Optional[str]): Additional context-specific details.
        potential_causes (Optional[str]): Possible reasons for the error.
    """

    def __init__(
        self, 
        message: str, 
        details: Optional[str] = None, 
        potential_causes: Optional[str] = None
    ):
        """Initialize a new MLE-Dojo Error.
        
        Args:
            message: Primary error message.
            details: Additional context-specific details.
            potential_causes: Possible reasons for the error. If not provided,
                extracted from class docstring if available.
        """
        self.message = message
        self.details = details
        
        # Extract potential causes from docstring if not provided
        if potential_causes is not None:
            self.potential_causes = potential_causes
        else:
            doc = self.__doc__ or ""
            self.potential_causes = self._extract_causes_from_docstring(doc)
        
        # Build complete error message
        error_msg = self._build_error_message()
        super().__init__(error_msg)
    
    def _extract_causes_from_docstring(self, docstring: str) -> Optional[str]:
        """Extract potential causes from class docstring.
        
        Args:
            docstring: The class docstring.
            
        Returns:
            Extracted potential causes or None if not found.
        """
        if "This typically occurs when:" in docstring:
            parts = docstring.split("This typically occurs when:")
            if len(parts) > 1:
                return parts[1].strip()
        return None
    
    def _build_error_message(self) -> str:
        """Build a complete error message with all available information.
        
        Returns:
            Formatted error message with message, details, and potential causes.
        """
        error_components = [self.message]
        
        if self.details:
            error_components.append(f"Details: {self.details}")
            
        if self.potential_causes:
            causes = self.potential_causes.replace("\n", "\n\t")
            error_components.append(f"Potential causes: \n\t{causes}")
            
        return "\n".join(error_components)


# ================================================
# Sandbox Related Errors
# ================================================

class SandboxError(MLEDojoError):
    """Base class for sandbox related errors.
    
    Groups all errors related to the execution sandbox environment.
    """
    pass


class SandboxInitializationError(SandboxError):
    """Raised when sandbox initialization fails.
    
    This typically occurs when:
    - Invalid resource limits specified
    - Log directory creation fails
    - GPU device not available
    """
    pass


class SandboxExecutionError(SandboxError):
    """Raised when code execution in sandbox fails.
    
    This typically occurs when:
    - Process creation fails
    - Resource limits exceeded
    - Execution timeout reached
    - Process termination fails
    """
    pass


class SandboxResourceError(SandboxError):
    """Raised when sandbox resource limits are exceeded.
    
    This typically occurs when:
    - Memory limit exceeded
    - CPU time limit exceeded 
    - GPU memory limit exceeded
    """
    pass


class SandboxCleanupError(SandboxError):
    """Raised when sandbox cleanup fails.
    
    This typically occurs when:
    - Process termination fails
    - Process group termination fails
    - Resource cleanup fails
    """
    pass


# ================================================
# Submission Related Errors
# ================================================

class SubmissionNotFoundError(MLEDojoError):
    """Raised when submission file is missing or cannot be found.
    
    This typically occurs when:
    - Code execution did not generate a submission file
    - Submission file was saved to wrong location
    - Submission file name does not match competition requirements
    """
    pass


class InvalidSubmissionError(MLEDojoError):
    """Raised when the submission cannot be graded.
    
    This typically occurs when:
    - Submission must be a pandas DataFrame
    - Number of rows doesn't match ground truth
    - First column values don't match between submission and ground truth
    - Missing required columns in submission
    - Extra unexpected columns in submission
    """
    pass

# ================================================
# Ground Truth Related Errors
# ================================================

class GroundTruthError(MLEDojoError):
    """Base class for ground truth related errors.
    
    This typically occurs when:
    - Ground truth file is missing
    - Ground truth file is corrupted/unreadable
    - Ground truth format is invalid
    """
    pass


# ================================================
# Validation Related Errors
# ================================================

class InvalidSubmissionError(MLEDojoError):
    """Raised when the submission is invalid.

    Refer to competition metrics for more details on validation requirements.
    
    This typically occurs when:
    - Submission format does not match requirements
    - Missing required fields in submission
    - Invalid values in submission
    """
    pass


# ================================================
# Evaluation Related Errors
# ================================================

class EvaluationError(MLEDojoError):
    """Raised during submission scoring/evaluation.
    
    This typically occurs when:
    - Metric calculation fails
    - Invalid values prevent score computation
    - Required evaluation data is missing
    """
    pass


# ================================================
# Leaderboard Related Errors
# ================================================

class LeaderboardError(MLEDojoError):
    """Raised during leaderboard operations.
    
    This typically occurs when:
    - Leaderboard file is missing/corrupted
    - Position calculation fails
    - Score comparison issues
    - Leaderboard update fails
    """
    pass


# ================================================
# Archive Related Errors
# ================================================

class ArchiveError(MLEDojoError):
    """Raised during submission archiving.
    
    This typically occurs when:
    - Insufficient permissions
    - Disk space issues
    - File system errors
    - Archive operation interrupted
    """
    pass


# ================================================
# Info Related Errors
# ================================================

class InfoError(MLEDojoError):
    """Raised during competition info retrieval.
    
    This typically occurs when:
    - Invalid info type requested
    - Info provider fails
    - Required files/paths not found
    - Configuration data is invalid
    """
    pass


# ================================================
# Code Related Errors
# ================================================

class CodeError(MLEDojoError):
    """Base class for code execution related errors.
    
    Groups all errors related to user code execution and analysis.
    """
    pass


class CodeSyntaxError(CodeError):
    """Raised when submitted code has syntax errors.
    
    This typically occurs when:
    - Python syntax is invalid
    - Indentation errors
    - Missing closing brackets/quotes
    - Import errors
    """
    pass


class CodeRuntimeError(CodeError):
    """Raised during code execution in sandbox.
    
    This typically occurs when:
    - Unhandled exceptions in code
    - Resource limits exceeded
    - Timeout reached
    - Dependencies missing
    """
    pass


# ================================================
# Competition Related Errors
# ================================================

class CompetitionError(MLEDojoError):
    """Base class for competition related errors.
    
    Groups all errors related to competition configuration and management.
    """
    pass


class CompetitionNotFoundError(CompetitionError):
    """Raised when requested competition does not exist.
    
    This typically occurs when:
    - Competition name not registered
    - Competition registry is empty
    - Competition was removed or archived
    """
    pass


class CompetitionAlreadyExistsError(CompetitionError):
    """Raised when registering duplicate competition.
    
    This typically occurs when:
    - Competition with same name already registered
    - Competition ID collision
    """
    pass


class CompetitionDataError(CompetitionError):
    """Raised when competition data directory is invalid.
    
    This typically occurs when:
    - Data directory does not exist
    - Missing required data files
    - Invalid data format
    - Corrupted data files
    """
    pass


class MetricError(CompetitionError):
    """Raised when competition metric class is invalid.
    
    This typically occurs when:
    - Metric class not provided
    - Metric is not a valid class type
    - Metric class does not inherit from CompetitionMetrics
    - Metric implementation is incomplete
    """
    pass


# ================================================
# Utility Functions
# ================================================

def error_to_dict(error: MLEDojoError) -> Dict[str, Any]:
    """Convert an MLEDojoError to a dictionary format.
    
    This function transforms an MLEDojoError instance into a standardized
    dictionary representation suitable for JSON serialization, logging,
    or API responses.
    
    Args:
        error: The MLEDojoError to convert.
        
    Returns:
        Dictionary representation of the error with status and details.
    """
    return {
        "status": "FAILED",
        "error_type": error.__class__.__name__,
        "error": error.message,
        "details": error.details,
        "potential_causes": error.potential_causes
    }


def create_error_response(
    error: Union[MLEDojoError, Exception]
) -> Dict[str, Any]:
    """Create a standardized error response.
    
    Handles both MLEDojoError instances and standard Python exceptions,
    converting them to a consistent dictionary format.
    
    Args:
        error: MLEDojoError or standard Exception.
        
    Returns:
        Dictionary containing error details.
    """
    if isinstance(error, MLEDojoError):
        return error_to_dict(error)
    
    # Handle standard exceptions
    return {
        "status": "FAILED",
        "error_type": error.__class__.__name__,
        "error": str(error),
        "details": None,
        "potential_causes": None
    }
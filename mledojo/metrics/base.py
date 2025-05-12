from abc import ABC, abstractmethod
from typing import Any

METRIC_CLASSES = list()


class CompetitionMetrics:
    """Abstract base class for competition evaluation metrics"""

    subclasses = []

    def __init__(self, higher_is_better: bool = True):
        self.higher_is_better = higher_is_better

    @abstractmethod
    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        """Evaluate predictions against ground truth"""
        pass

    @abstractmethod
    def validate_submission(self, submission: Any, ground_truth: Any) -> tuple[bool, str]:
        """Validate submission format and contents"""
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CompetitionMetrics.subclasses.append(cls)


class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass

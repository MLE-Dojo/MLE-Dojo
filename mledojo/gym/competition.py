"""
Competition Management System for Machine Learning Competitions.

This module provides a framework for managing machine learning competitions,
including registration, data management, and metric evaluation. It implements
a registry pattern for tracking multiple competitions and provides structured
access to competition data and evaluation metrics.

Typical usage:
    registry = CompetitionRegistry()
    registry.register(
        name="my_competition",
        data_dir="/path/to/data",
        comp_info=CompInfo(category="Vision", level="advanced"),
        metric_class=MyCustomMetrics
    )
    competition = registry.get("my_competition")
    metrics = competition.create_metrics()
"""

from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from pathlib import Path
import logging
from mledojo.metrics.base import CompetitionMetrics
from mledojo.gym.error import CompetitionNotFoundError, CompetitionAlreadyExistsError, CompetitionDataError, MetricError

logger = logging.getLogger(__name__)


@dataclass
class CompInfo:
    """
    Competition metadata and configuration information.

    Attributes:
        category: The competition category (e.g., "Tabular", "Vision", "NLP")
        level: Difficulty level of the competition (e.g., "beginner", "intermediate", "advanced")
        output_type: Expected submission file format (e.g., "submission.csv")
        higher_is_better: Whether higher metric values indicate better performance
        tldr: Optional one-liner description of the competition
        tags: Optional list of tags associated with this competition
    """

    category: str = "Tabular"
    level: str = "intermediate"
    output_type: str = "submission.csv"
    higher_is_better: bool = True
    tldr: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate competition information after initialization."""
        self.level = self.level.lower()
        if self.level not in ["beginner", "intermediate", "advanced"]:
            logger.warning(
                f"Unusual competition level: '{self.level}'. "
                f"Common values are 'beginner', 'intermediate', or 'advanced'."
            )
        if self.output_type != "submission.csv":
            logger.warning(f"Unusual output type: '{self.output_type}'. " f"Common values are 'submission.csv'.")


class Competition:
    """
    Class representing a competition with its data, information, and evaluation metrics.

    A competition has a unique name, a data directory with a specific structure,
    metadata about the competition type and goals, and a metrics class for evaluation.

    Attributes:
        name: Unique identifier for the competition
        data_dir: Path to competition data directory
        info: Competition metadata and configuration
        metric_class: Class implementing competition evaluation metrics
    """

    # Required directory structure for competitions
    REQUIRED_PATHS = [
        "",  # Base directory
        "public",
        "private",
        "public/description.txt",
        "public/sample_submission.csv",
        "private/test_answer.csv",
    ]

    def __init__(
        self,
        name: str,
        data_dir: str,
        comp_info: Optional[CompInfo] = None,
        metric_class: Optional[Type[CompetitionMetrics]] = None,
    ):
        """
        Initialize a competition with its data and evaluation metrics.

        Args:
            name: Unique identifier for the competition
            data_dir: Directory containing competition data
            comp_info: Competition metadata and configuration
            metric_class: Class implementing competition evaluation metrics

        Raises:
            MetricError: If metric_class is invalid or None
            CompetitionDataError: If required data structure is missing
        """
        self.name = name
        self.data_dir = str(data_dir)  # Convert to string for consistency
        self.info = comp_info or CompInfo()

        # Validate metric class
        if metric_class is None:
            raise MetricError(message="Metric class not provided", details="Competition requires a valid metric class")

        if not issubclass(metric_class, CompetitionMetrics):
            raise MetricError(
                message="Invalid metric class", details="Metric class must inherit from CompetitionMetrics"
            )

        self.metric_class = metric_class

        # Validate data directory structure
        self._validate_data_structure()

    def _validate_data_structure(self) -> None:
        """
        Validate that the competition data directory has the required structure.

        Raises:
            CompetitionDataError: If any required path is missing
        """
        data_path = Path(self.data_dir)

        for relative_path in self.REQUIRED_PATHS:
            path = data_path / relative_path if relative_path else data_path
            if not path.exists():
                raise CompetitionDataError(
                    message=f"Required path does not exist: {path}",
                    details="Competition requires specific directory structure and files",
                )

    def get_data_path(self) -> Path:
        """
        Return the data directory as a Path object.

        Returns:
            Path object pointing to the competition data directory
        """
        return Path(self.data_dir)

    def get_public_data_path(self) -> Path:
        """
        Return the public data directory as a Path object.

        Returns:
            Path object pointing to the public competition data
        """
        return self.get_data_path() / "public"

    def get_private_data_path(self) -> Path:
        """
        Return the private data directory as a Path object.

        Returns:
            Path object pointing to the private competition data
        """
        return self.get_data_path() / "private"

    def create_metrics(self) -> CompetitionMetrics:
        """
        Create an instance of the competition's metric class.

        Returns:
            An instance of the competition's metric class

        Raises:
            MetricError: If metrics creation fails
        """
        try:
            return self.metric_class()
        except Exception as e:
            raise MetricError(message="Failed to create metrics instance", details=str(e))

    def __str__(self) -> str:
        """Return string representation of the competition."""
        return f"Competition(name='{self.name}', category='{self.info.category}', level='{self.info.level}')"

    def __repr__(self) -> str:
        """Return detailed string representation of the competition."""
        return (
            f"Competition(name='{self.name}', data_dir='{self.data_dir}', "
            f"category='{self.info.category}', level='{self.info.level}', "
            f"metric_class={self.metric_class.__name__})"
        )


class CompetitionRegistry:
    """
    Registry for managing multiple competitions.

    Provides methods for registering, retrieving, and listing competitions.
    Each competition has a unique name within the registry.
    """

    def __init__(
        self,
        name: str = None,
        data_dir: str = None,
        comp_info: CompInfo = None,
        metric_class: Type[CompetitionMetrics] = None,
        competitions: dict[str, Competition] = None,
    ):
        """
        Initialize the class by:
        - Registering a placeholder with the actual initialization conducted later through `register()`
        - Or registering a competition with the provided name, data_dir, comp_info, and metric_class
        - Or using an existing dictionary of competitions.

        """
        if competitions is not None and isinstance(competitions, dict):
            # If competitions is provided, use it to initialize the registry
            self._competitions = competitions
            return

        self._competitions: Dict[str, Competition] = {}

        if name is not None:
            # If name is provided, register a default competition
            self.register(name=name, data_dir=data_dir, comp_info=comp_info, metric_class=metric_class)

    def register(
        self,
        name: str,
        data_dir: str,
        comp_info: Optional[CompInfo] = None,
        metric_class: Optional[Type[CompetitionMetrics]] = None,
    ) -> Competition:
        """
        Register a competition with its info and metric class.

        Args:
            name: Unique identifier for the competition
            data_dir: Directory containing competition data
            comp_info: Competition metadata and configuration
            metric_class: Class implementing competition evaluation metrics

        Returns:
            The newly created Competition object

        Raises:
            CompetitionAlreadyExistsError: If competition name already exists
            MetricError: If metric_class is invalid
            CompetitionDataError: If required data structure is missing
        """
        if name in self._competitions:
            raise CompetitionAlreadyExistsError(
                message=f"Competition '{name}' already registered", details="Each competition must have a unique name"
            )

        competition = Competition(name=name, data_dir=data_dir, comp_info=comp_info, metric_class=metric_class)

        self._competitions[name] = competition
        logger.info(f"Registered competition '{name}'")
        return competition

    def get(self, name: str) -> Competition:
        """
        Retrieve competition by name.

        Args:
            name: Name of competition to retrieve

        Returns:
            Competition object

        Raises:
            CompetitionNotFoundError: If competition not found
        """
        if name not in self._competitions:
            raise CompetitionNotFoundError(
                message=f"Competition '{name}' not found",
                details="Competition must be registered before it can be retrieved",
            )
        return self._competitions[name]

    def unregister(self, name: str) -> None:
        """
        Remove a competition from the registry.

        Args:
            name: Name of competition to remove

        Raises:
            CompetitionNotFoundError: If competition not found
        """
        if name not in self._competitions:
            raise CompetitionNotFoundError(
                message=f"Competition '{name}' not found", details="Cannot unregister a competition that doesn't exist"
            )
        del self._competitions[name]
        logger.info(f"Unregistered competition '{name}'")

    def list_competitions(self) -> List[str]:
        """
        Get list of all registered competition names.

        Returns:
            List of competition names
        """
        return list(self._competitions.keys())

    def get_competitions_by_category(self, category: str) -> List[Competition]:
        """
        Get list of competitions that match the specified category.

        Args:
            category: Category to filter by (e.g., "Tabular", "Vision")

        Returns:
            List of matching Competition objects
        """
        return [comp for comp in self._competitions.values() if comp.info.category.lower() == category.lower()]

    def get_competitions_by_level(self, level: str) -> List[Competition]:
        """
        Get list of competitions that match the specified difficulty level.

        Args:
            level: Level to filter by (e.g., "beginner", "intermediate", "advanced")

        Returns:
            List of matching Competition objects
        """
        return [comp for comp in self._competitions.values() if comp.info.level.lower() == level.lower()]

    def __contains__(self, name: str) -> bool:
        """
        Check if competition exists in the registry.

        Args:
            name: Name of competition to check

        Returns:
            True if competition exists, False otherwise
        """
        return name in self._competitions

    def __len__(self) -> int:
        """
        Get the number of registered competitions.

        Returns:
            Number of competitions in the registry
        """
        return len(self._competitions)

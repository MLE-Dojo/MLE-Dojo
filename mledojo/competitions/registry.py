# mledojo/competitions/registry.py

import os
import importlib.util
import importlib.resources # Preferred way to access package data
import inspect
import sys
from pathlib import Path
from typing import Optional, Type, Callable, Dict, Any, List
from functools import lru_cache

# --- Locate the base 'competitions' directory reliably ---
try:
    # Use importlib.resources to find the path relative to the installed package
    _pkg_files = importlib.resources.files('mledojo')
    COMPETITIONS_DIR_PATH = Path(str(_pkg_files / 'competitions')).resolve()
except ImportError:
    # Fallback for running directly from source code (less robust)
    print("Warning: Could not use importlib.resources. Trying fallback path finding.", file=sys.stderr)
    _this_file = Path(__file__).resolve() # Path to this registry.py file
    COMPETITIONS_DIR_PATH = _this_file.parent # Assumes registry.py is in competitions/
    # Verify the expected parent structure exists
    if not (COMPETITIONS_DIR_PATH.parent / '__init__.py').exists() or \
       not (COMPETITIONS_DIR_PATH / '__init__.py').exists():
         raise ImportError(f"Fallback path finding failed: Cannot reliably locate mledojo/competitions directory from {COMPETITIONS_DIR_PATH}.")

if not COMPETITIONS_DIR_PATH.is_dir():
    print(f"Error: Competitions directory not found at resolved path: {COMPETITIONS_DIR_PATH}", file=sys.stderr)
    # Depending on requirements, you might raise an error or just have an empty registry
    # raise EnvironmentError(f"Competitions directory missing: {COMPETITIONS_DIR_PATH}")


class CompetitionInfo:
    """Holds resolved paths for a discovered competition."""
    def __init__(self, name: str, base_path: Path):
        self.name: str = name # Original name with hyphens
        self.base_path: Path = base_path
        self.utils_path: Path = base_path / 'utils'
        self.info_path: Path = base_path / 'info'
        self.metric_py_path: Path = self.utils_path / 'metric.py'
        self.prepare_py_path: Path = self.utils_path / 'prepare.py'
        self.description_txt_path: Path = self.info_path / 'description.txt'
        self.public_leaderboard_csv_path: Path = self.info_path / 'public_leaderboard.csv'
        self.private_leaderboard_csv_path: Path = self.info_path / 'private_leaderboard.csv'

    def __repr__(self):
        return f"<CompetitionInfo(name='{self.name}')>"

# --- Registry Dictionary ---
_registry: Dict[str, CompetitionInfo] = {}
_discovered = False

def _discover_competitions():
    """Scans the competitions directory and populates the registry."""
    global _discovered, _registry
    if _discovered or not COMPETITIONS_DIR_PATH.is_dir():
        return

    _registry = {}
    for item in COMPETITIONS_DIR_PATH.iterdir():
        # Check if it's a directory, not hidden, not __pycache__, and seems like a competition
        if item.is_dir() and not item.name.startswith(('.', '__')):
            comp_name = item.name # Keep the original name with hyphens
            utils_dir = item / 'utils'
            metric_py = utils_dir / 'metric.py'
            # Add more validation if needed (e.g., check for info dir)
            if utils_dir.is_dir() and metric_py.is_file():
                _registry[comp_name] = CompetitionInfo(name=comp_name, base_path=item)

    _discovered = True
    # print(f"Discovered {len(_registry)} competitions in {COMPETITIONS_DIR_PATH}") # Optional debug


def get_competition_info(competition_name: str) -> Optional[CompetitionInfo]:
    """Gets the CompetitionInfo object for a given competition name."""
    if not _discovered:
        _discover_competitions()
    info = _registry.get(competition_name)
    if not info:
         print(f"Warning: Competition '{competition_name}' not found in registry.", file=sys.stderr)
    return info

def list_competitions() -> List[str]:
    """Returns a list of discovered competition names."""
    if not _discovered:
        _discover_competitions()
    return sorted(list(_registry.keys()))

@lru_cache(maxsize=128) # Cache recently loaded modules
def _load_module_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Dynamically loads a Python module directly from its file path."""
    if not file_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
             print(f"Error: Could not create module spec for: {file_path}", file=sys.stderr)
             return None
        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules BEFORE execution for potential relative imports within the module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {module_name} from {file_path}: {e}", file=sys.stderr)
        if module_name in sys.modules: # Clean up if loading failed
            del sys.modules[module_name]
        return None

# def get_metric(competition_name: str) -> Optional[Type]:
#     """Gets the metric class for a specific competition."""
#     info = get_competition_info(competition_name)
#     if not info or not info.metric_py_path.is_file():
#         return None

#     # Create a unique but valid module name for importlib internal use
#     internal_module_name = f"mledojo._loaded_competitions.{info.name.replace('-', '_')}.utils.metric"
#     module = _load_module_from_path(internal_module_name, info.metric_py_path)

#     if module:
#         for name, obj in inspect.getmembers(module):
#             # Adjust class check logic as needed (e.g., based on inheritance)
#             if inspect.isclass(obj) and name != 'CompetitionMetrics':
#                 return obj
#         print(f"Warning: No suitable '*Metrics*' class found in {info.metric_py_path}", file=sys.stderr)
#     return None

def get_metric(competition_name: str) -> Optional[Type]:
    """Gets the metric class for a specific competition."""
    info = get_competition_info(competition_name)
    if not info or not info.metric_py_path.is_file():
        return None

    import mledojo.metrics.base as base_module
    sys.modules['mledojo.metrics.base'] = base_module
    from mledojo.metrics.base import CompetitionMetrics

    # Create a unique but valid module name for importlib internal use
    internal_module_name = f"mledojo._loaded_competitions.{info.name.replace('-', '_')}.utils.metric"
    module = _load_module_from_path(internal_module_name, info.metric_py_path)

    if module:
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and name != 'CompetitionMetrics'
                and issubclass(obj, CompetitionMetrics)
            ):
                return obj

        print(f"Warning: No suitable '*Metrics*' class found in {info.metric_py_path}", file=sys.stderr)
    return None

def get_prepare(competition_name: str) -> Optional[Callable]:
    """Gets the prepare function for a specific competition."""
    info = get_competition_info(competition_name)
    if not info or not info.prepare_py_path.is_file():
        return None # Prepare might be optional

    internal_module_name = f"mledojo._loaded_competitions.{info.name.replace('-', '_')}.utils.prepare"
    module = _load_module_from_path(internal_module_name, info.prepare_py_path)

    if module:
        prepare_func = getattr(module, 'prepare', None)
        if inspect.isfunction(prepare_func):
            return prepare_func
        print(f"Warning: No function named 'prepare' found in {info.prepare_py_path}", file=sys.stderr)
    return None

def get_competition_description(competition_name: str) -> Optional[str]:
    """Gets the description text content for a competition."""
    info = get_competition_info(competition_name)
    if not info or not info.description_txt_path.is_file():
        return None
    try:
        return info.description_txt_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading description for {competition_name}: {e}", file=sys.stderr)
        return None

# Add similar functions to get paths or content for leaderboards etc.
def get_public_leaderboard_path(competition_name: str) -> Optional[Path]:
    info = get_competition_info(competition_name)
    if info and info.public_leaderboard_csv_path.is_file():
        return info.public_leaderboard_csv_path
    return None
"""
Utility functions for the MLE-Dojo framework.

This module provides various utility functions for the MLE-Dojo framework, including:
- Competition-related utilities (metric and preparation functions)
- Configuration management (loading and creating configs)
- Process management (handling timeouts and terminating processes)
- Environment setup (creating and configuring Kaggle environments)
- Agent history management (setting up directories for agent outputs)

These utilities are used throughout the framework to support the execution
of machine learning agents in Kaggle competition environments.
"""

import os
import importlib
import inspect
import yaml
import psutil
import threading
import json
from typing import Optional, Type, Any, Dict, List, Callable, Tuple
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from rich.console import Console
from rich import print as rprint
from mledojo.competitions import get_metric

# ============================================================================
# Competition-related utilities
# ============================================================================

# def get_metric(competition_name: str) -> Optional[Type]:
#     """
#     Get the metric class for a specific competition.
    
#     Dynamically imports the metric class from the competition's module.
#     Looks for classes ending with 'Metrics' that are not the base class.
    
#     Args:
#         competition_name: The name of the competition
        
#     Returns:
#         The metric class for the competition if found, None otherwise
#     """
#     module_path = f"mledojo.competitions.{competition_name}.utils.metric"
    
#     try:
#         module = importlib.import_module(module_path)
        
#         for name, obj in inspect.getmembers(module):
#             if inspect.isclass(obj) and name.endswith('Metrics') and name != 'CompetitionMetrics':
#                 return obj
        
#         return None
#     except Exception as e:
#         print(f"Error importing metrics for {competition_name}: {e}")
#         return None


# def get_prepare(competition_name: str) -> Optional[Callable]:
#     """
#     Get the prepare function for a specific competition.
    
#     Dynamically imports the prepare function from the competition's module.
    
#     Args:
#         competition_name: The name of the competition
    
#     Returns:
#         The prepare function for the competition if found, None otherwise
#     """
#     module_path = f"mledojo.competitions.{competition_name}.utils.prepare"
    
#     try:
#         module = importlib.import_module(module_path)
        
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj) and name == 'prepare':
#                 return obj
        
#         return None
#     except Exception as e:
#         print(f"Error importing prepare function for {competition_name}: {e}")
#         return None


# ============================================================================
# Configuration management
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['agent'] = load_agent_config(config['agent_type'])
    return config


def load_agent_config(agent_type: str) -> Dict[str, Any]:
    """
    Load agent-specific configuration from the appropriate config file.
    
    Args:
        agent_type: Type of agent (aide, mle, openai, dummy)
        
    Returns:
        Agent configuration dictionary
    """
    config_paths = {
        'aide': './mledojo/agent/aide/utils/config.yaml',
        'mle': './mledojo/agent/mleagent/config.yaml',
        'openai': './mledojo/agent/openaiagent/config.yaml',
        'dummy': './mledojo/agent/dummy/config.yaml'
    }
    
    if agent_type not in config_paths:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be one of: {', '.join(config_paths.keys())}")
    
    config_path = config_paths[agent_type]
    
    try:
        with open(config_path, 'r') as f:
            agent_config = yaml.safe_load(f)
            return agent_config.get('agent', {})
    except FileNotFoundError:
        rprint(f"[bold red]Warning:[/bold red] Agent config file not found at {config_path}. Using empty config.")
        return {}


def create_config_from_args(args: Any) -> Dict[str, Any]:
    """
    Create a configuration dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'output_dir': args.output_dir,
        'competition': {
            'name': args.competition_name,
            'data_dir': args.data_dir
        },
        'env': {
            'render_mode': args.render_mode,
            'gpu_device': args.gpu_device,
            'gpu_memory_limit': args.gpu_memory_limit,
            'execution_timeout': args.execution_timeout,
            'score_mode': args.score_mode,
            'max_steps': args.max_steps,
        },
        'agent_type': args.agent_type,
        'agent': {},
    }
    
    # Load agent-specific configuration from the appropriate config file
    agent_config = load_agent_config(args.agent_type)
    config['agent'].update(agent_config)
    
    return config


# ============================================================================
# Process management
# ============================================================================

def get_all_child_processes(parent_pid: int) -> List[psutil.Process]:
    """
    Get all child processes of a given parent process.
    
    Args:
        parent_pid: Parent process ID
        
    Returns:
        List of child process objects
    """
    try:
        parent = psutil.Process(parent_pid)
        return parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def terminate_process_and_children(pid: int) -> None:
    """
    Terminate a process and all its children.
    
    First attempts to terminate processes gracefully, then forcibly kills
    any processes that don't terminate within the timeout.
    
    Args:
        pid: Process ID to terminate
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # First terminate children
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Give children time to terminate
        _, still_alive = psutil.wait_procs(children, timeout=3)
        
        # Kill any remaining children
        for proc in still_alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Finally terminate the parent
        parent.terminate()
        parent.wait(3)
        
        if parent.is_running():
            parent.kill()
            
    except psutil.NoSuchProcess:
        pass


def timeout_handler(timeout: int, func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Run a function with a timeout. If the function doesn't complete within
    the timeout, terminate it and all its child processes.
    
    Args:
        timeout: Timeout in seconds
        func: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function or None if timeout occurred
    """
    result = [None]
    exception = [None]
    completed = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    thread.start()
    thread.join(timeout)
    
    # If the thread is still alive after the timeout
    if thread.is_alive():
        print(f"Timeout reached after {timeout} seconds. Terminating function execution.")
        
        # Get the current process and all its children
        current_pid = os.getpid()
        child_processes = get_all_child_processes(current_pid)
        
        print(f"Found {len(child_processes)} child processes to terminate.")
        
        # Terminate all child processes
        for proc in child_processes:
            try:
                print(f"Terminating process {proc.pid} ({proc.name()})")
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Give processes time to terminate
        _, still_alive = psutil.wait_procs(child_processes, timeout=3)
        
        # Kill any remaining processes
        if still_alive:
            print(f"{len(still_alive)} processes still alive after termination, killing forcibly.")
            for proc in still_alive:
                try:
                    print(f"Killing process {proc.pid} forcibly")
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        
        # Use terminate_process_and_children to ensure all processes are terminated
        for proc in child_processes:
            try:
                terminate_process_and_children(proc.pid)
            except Exception as e:
                print(f"Error terminating process {proc.pid}: {e}")
        
        # Raise an exception to interrupt the main thread's execution
        # This will help prevent the function from continuing to run in the background
        raise TimeoutError(f"Function execution timed out after {timeout} seconds")
    
    # If an exception occurred in the thread, raise it
    if exception[0] is not None:
        raise exception[0]
    
    return result[0]


# ============================================================================
# Environment setup
# ============================================================================

def setup_environment(config: Dict[str, Any]) -> KaggleEnvironment:
    """
    Set up the Kaggle environment based on configuration.
    
    Creates and configures a KaggleEnvironment instance with the appropriate
    competition registry, rendering mode, and resource limits.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured KaggleEnvironment instance
    """
    competition_name = config['competition']['name']
    output_dir = config['output_dir']
    data_dir = config['competition']['data_dir']
    env_config = config['env']
    
    competition_registry = CompetitionRegistry()
    comp_info = CompInfo()
    metric_class = get_metric(competition_name)
    
    competition_registry.register(
        competition_name, 
        data_dir=data_dir, 
        comp_info=comp_info, 
        metric_class=metric_class
    )
    
    env = KaggleEnvironment(
        competition_name=competition_name,
        output_dir=output_dir,
        competition_registry=competition_registry,
        render_mode=env_config.get('render_mode', 'human'),
        gpu_device=env_config.get('gpu_device', None),
        gpu_memory_limit=env_config.get('gpu_memory_limit', None),
        execution_timeout=env_config.get('execution_timeout', 43200),
        score_mode=env_config.get('score_mode', 'position')
    )
    
    return env


# ============================================================================
# Agent history management
# ============================================================================

def setup_agent_history_dir(output_dir: str) -> Tuple[str, str, str, str]:
    """
    Set up directories and files for agent history.
    
    Creates the necessary directories and returns file paths for storing
    agent trajectory, error fixing history, and cost tracking.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Tuple of (agent_history_dir, trajectory_file, fix_parse_error_file, cost_history_file)
    """
    agent_history_dir = os.path.join(output_dir, "agent_history")
    os.makedirs(agent_history_dir, exist_ok=True)
    
    trajectory_file = os.path.join(agent_history_dir, "trajectory.json")
    fix_parse_error_file = os.path.join(agent_history_dir, "fix_parse_error.json")
    cost_history_file = os.path.join(agent_history_dir, "cost_history.json")
    
    return agent_history_dir, trajectory_file, fix_parse_error_file, cost_history_file
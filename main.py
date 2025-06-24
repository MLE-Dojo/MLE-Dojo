"""
Main script for running Kaggle competition agents.

This script provides a unified interface for running different types of agents
(MLE, OpenAI, AIDE, Dummy) on Kaggle competitions. It handles configuration loading,
environment setup, agent initialization, and execution with proper timeout handling.

Usage:
    python main.py --config path/to/config.yaml
    or
    python main.py with direct parameters (see argument parser for details)

Examples:
    python main.py --config configs/mle_agent_config.yaml
    python main.py --agent-type mle --competition-name connectx --data-dir data/connectx --output-dir output
"""

import os
import json
import argparse
import psutil
import asyncio
import time
import logging
import sys  # Import sys for stdout configuration
from typing import Dict, Any, Tuple
from pathlib import Path
# Agent imports
from mledojo.agent.aide.utils.config import save_run
from mledojo.agent.mleagent.buildup import setup_mle_agent
from mledojo.agent.openaiagent.buildup import setup_openai_agent
from mledojo.agent.aide.buildup import setup_aide_agent
from mledojo.agent.dummy.buildup import setup_dummy_agent

# Environment imports
from mledojo.gym.env import KaggleEnvironment
from mledojo.utils import (
    load_config,
    create_config_from_args,
    get_all_child_processes,
    timeout_handler,
    setup_environment,
    setup_agent_history_dir,
    load_agent_config
)
from rich.console import Console
from rich import print as rprint

# --- Centralized Logger Setup ---
# Configure the main logger for the application
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Log to standard output
log_handler.setFormatter(log_formatter)

# Get the main logger instance
logger = logging.getLogger('mledojo_main')
logger.setLevel(logging.INFO) # Set desired log level
logger.addHandler(log_handler)
# Prevent messages from propagating to the root logger if it has handlers
logger.propagate = False 
# --- End Logger Setup ---


async def run_openai_agent(
    agent: Any,
    env: KaggleEnvironment,
    output_files: Tuple[str, str, str],
    config: Dict[str, Any]
) -> None:
    """
    Run the OpenAI agent asynchronously.
    
    Args:
        agent: The OpenAI agent instance
        env: The Kaggle environment
        output_files: Tuple of file paths for saving agent history
                     (trajectory_file, fix_parse_error_file, cost_history_file)
        config: Configuration dictionary containing execution parameters
    """
    trajectory_file, fix_parse_error_file, cost_history_file = output_files
    action_left = config['env']['max_steps']
    execution_timeout = config['env']['execution_timeout']
    obs = None

    # Log model and task information
    model_name = config['agent'].get('model_name', 'Unknown model')
    competition_name = config['competition']['name']

    logger.info(f"Starting OpenAI agent run. Model: {model_name}, Competition: {competition_name}")
    logger.info(f"Max steps: {action_left}, Timeout: {execution_timeout}s")

    start_time = time.time()

    for i in range(action_left):
        step_num = i + 1
        logger.info(f"--- Step {step_num}/{action_left} ---") # Log step start
        
        # Calculate remaining time
        elapsed_time = time.time() - start_time
        time_left = max(0, execution_timeout - elapsed_time)
        
        action, params = await agent.act(obs, action_left - i, int(time_left))
        logger.info(f"Step {step_num}: Action received: {action}")

        if action == "Error":
            logger.error(f"Step {step_num}: Agent returned error: {params}")
            break
        if action == "End":
            logger.warning(f"Step {step_num}: Agent indicated end (e.g., token limit reached).")
            break

        # Write conversation history to files
        with open(trajectory_file, 'w') as f:
            json.dump(agent.history_to_save, f, indent=4)
        
        with open(fix_parse_error_file, 'w') as f:
            json.dump(agent.fix_parse_history, f, indent=4)
        
        with open(cost_history_file, 'w') as f:
            json.dump(agent.cost_history, f, indent=4)
            
        obs, reward = env.step(action, **params)
        logger.info(f"Step {step_num}: Environment step executed. Reward: {reward}")
        
        # Check if time limit exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time >= execution_timeout:
            logger.warning(f"Time limit ({execution_timeout}s) exceeded after step {step_num}.")
            break
            
    logger.info("OpenAI agent run finished.")


def run_mle_agent(
    agent: Any,
    env: KaggleEnvironment,
    output_files: Tuple[str, str, str],
    config: Dict[str, Any]
) -> None:
    """
    Run the MLE agent.
    
    Args:
        agent: The MLE agent instance
        env: The Kaggle environment
        output_files: Tuple of file paths for saving agent history
                     (trajectory_file, fix_parse_error_file, cost_history_file)
        config: Configuration dictionary containing execution parameters
    """
    trajectory_file, fix_parse_error_file, cost_history_file = output_files
    action_left = config['env']['max_steps']
    execution_timeout = config['env']['execution_timeout']
    obs = None

    # Log model and task information
    model_name = config['agent'].get('model_name', 'Unknown model')
    competition_name = config['competition']['name']

    logger.info(f"Starting MLE agent run. Model: {model_name}, Competition: {competition_name}")
    logger.info(f"Max steps: {action_left}, Timeout: {execution_timeout}s")

    start_time = time.time()

    for i in range(action_left):
        step_num = i + 1
        logger.info(f"--- Step {step_num}/{action_left} ---") # Log step start
        
        # Calculate remaining time
        elapsed_time = time.time() - start_time
        time_left = max(0, execution_timeout - elapsed_time)
        
        action, params = agent.act(obs, action_left - i, int(time_left))
        logger.info(f"Step {step_num}: Action received: {action}")

        if action == "Error":
            logger.error(f"Step {step_num}: Agent returned error: {params}")
            break
        if action == "End":
            logger.warning(f"Step {step_num}: Agent indicated end (e.g., token limit reached).")
            break

        # Write conversation history to files
        with open(trajectory_file, 'w') as f:
            json.dump(agent.conversation_history, f, indent=4)
        
        with open(fix_parse_error_file, 'w') as f:
            json.dump(agent.fix_parse_history, f, indent=4)
        
        with open(cost_history_file, 'w') as f:
            json.dump(agent.cost_history, f, indent=4)
            
        obs, reward = env.step(action, **params)
        logger.info(f"Step {step_num}: Environment step executed. Reward: {reward}")
        
        # Check if time limit exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time >= execution_timeout:
            logger.warning(f"Time limit ({execution_timeout}s) exceeded after step {step_num}.")
            break
            
    logger.info("MLE agent run finished.")


def run_aide_agent(
    agent: Any,
    env: KaggleEnvironment,
    journal: Any,
    cfg: Any,
    config: Dict[str, Any]
) -> None:
    """
    Run the AIDE agent.
    
    Args:
        agent: The AIDE agent instance
        env: The Kaggle environment
        journal: The AIDE journal for tracking experiments
        cfg: The AIDE configuration object
        config: Main configuration dictionary
    """
    steps = cfg.agent.get('steps', 20)
    execution_timeout = config['env']['execution_timeout']

    # Log model and task information
    model_name = cfg.agent.code.get('model_name', 'Unknown model')
    competition_name = config['competition']['name']

    logger.info(f"Starting AIDE agent run. Model: {model_name}, Competition: {competition_name}")
    logger.info(f"Total steps: {steps}, Timeout: {execution_timeout}s")

    start_time = time.time()

    for i in range(steps):
        step_num = i + 1
        logger.info(f"--- AIDE Step {step_num}/{steps} ---") # Log step start
        
        # Check if time limit exceeded before starting step
        elapsed_time = time.time() - start_time
        if elapsed_time >= execution_timeout:
            logger.warning(f"Time limit ({execution_timeout}s) exceeded before step {step_num}.")
            break
        
        # Create a wrapper function to adapt the gym._handle_execute_code output format
        def exec_callback(code: str) -> Any:
            """Execute code in the environment and return observation."""
            obs, reward = env.step("execute_code", **{"code": code})
            logger.info(f"Reward: {reward}")
            # Transform the result and reward into the expected format for the agent
            return obs
        
        agent.step(exec_callback=exec_callback)
        save_run(cfg, journal)

    best_node = journal.get_best_node(only_good=False)
    logger.info(f"AIDE agent run finished. Best solution metric: {best_node.metric.value}")


def run_dummy_agent(
    agent: Any,
    env: KaggleEnvironment,
    output_files: Tuple[str, str, str],
    config: Dict[str, Any]
) -> None:
    """
    Run the Dummy agent.
    
    Args:
        agent: The Dummy agent instance
        env: The Kaggle environment
        output_files: Tuple of file paths for saving agent history
                     (trajectory_file, fix_parse_error_file, cost_history_file)
        config: Configuration dictionary containing execution parameters
    """
    trajectory_file, fix_parse_error_file, cost_history_file = output_files
    action_left = config['env']['max_steps']
    execution_timeout = config['env']['execution_timeout']
    obs = None

    # Log model and task information
    competition_name = config['competition']['name']

    logger.info(f"Starting Dummy agent run. Competition: {competition_name}")
    logger.info(f"Max steps: {action_left}, Timeout: {execution_timeout}s")

    start_time = time.time()

    for i in range(action_left):
        step_num = i + 1
        logger.info(f"--- Step {step_num}/{action_left} ---") # Log step start
        
        # Check if time limit exceeded before starting step
        elapsed_time = time.time() - start_time
        if elapsed_time >= execution_timeout:
            logger.warning(f"Time limit ({execution_timeout}s) exceeded before step {step_num}.")
            break
        
        action, params = agent.act(obs)
        logger.info(f"Step {step_num}: Action received: {action}")

        # Write history to files if the agent supports it
        if hasattr(agent, 'conversation_history'):
            with open(trajectory_file, 'w') as f:
                json.dump(agent.conversation_history, f, indent=4)
        
        if hasattr(agent, 'fix_parse_history'):
            with open(fix_parse_error_file, 'w') as f:
                json.dump(agent.fix_parse_history, f, indent=4)
        
        if hasattr(agent, 'cost_history'):
            with open(cost_history_file, 'w') as f:
                json.dump(agent.cost_history, f, indent=4)  

        obs, reward = env.step(action, **params)
        logger.info(f"Step {step_num}: Environment step executed. Reward: {reward}")
        
    logger.info("Dummy agent run finished.")


async def run_openai_with_timeout(
    agent: Any,
    env: KaggleEnvironment,
    output_files: Tuple[str, str, str],
    config: Dict[str, Any],
    timeout: int
) -> None:
    """
    Run OpenAI agent with timeout handling using asyncio.
    
    Args:
        agent: The OpenAI agent instance
        env: The Kaggle environment
        output_files: Tuple of file paths for saving agent history
        config: Configuration dictionary
        timeout: Timeout in seconds
    """
    try:
        logger.info(f"Executing OpenAI agent with timeout: {timeout} seconds")
            
        await asyncio.wait_for(
            run_openai_agent(agent, env, output_files, config),
            timeout=timeout
        )
        logger.info("OpenAI agent finished within timeout.")
    except asyncio.TimeoutError:
        logger.error(f"Timeout reached ({timeout}s) during OpenAI agent execution. Terminating related processes.")
            
        # Get the current process and all its children
        current_pid = os.getpid()
        child_processes = get_all_child_processes(current_pid)
        
        # Terminate all child processes
        for proc in child_processes:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Give processes time to terminate
        _, still_alive = psutil.wait_procs(child_processes, timeout=3)
        
        # Kill any remaining processes
        for proc in still_alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


def main() -> None:
    """
    Main function to run the agent on a Kaggle competition.
    
    This function:
    1. Parses command line arguments
    2. Loads configuration from file or creates it from arguments
    3. Sets up the appropriate agent based on the configuration
    4. Sets up the Kaggle environment
    5. Runs the agent with proper timeout handling
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run an agent on a Kaggle competition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Direct parameter options from config.yaml
    parser.add_argument("--output-dir", type=str, help="Output directory for agent results and logs")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode")
    parser.add_argument("--competition-name", type=str, help="Name of the Kaggle competition")
    parser.add_argument("--data-dir", type=str, help="Directory containing competition data")
    parser.add_argument("--gpu-device", type=lambda x: int(x) if x.lower() != "null" else None, default=0, help="GPU device index")
    parser.add_argument("--gpu-memory-limit", type=lambda x: int(x) if x.lower() != "null" else None, default=32, help="GPU memory limit in GB")
    parser.add_argument("--execution-timeout", type=int, default=43200, help="Execution timeout in seconds")
    parser.add_argument("--score-mode", type=str, default="position", help="Score mode")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of steps")
    
    # Agent type parameter
    parser.add_argument(
        "--agent-type", 
        type=str, 
        choices=["mle", "openai", "aide", "dummy"], 
        help="Type of agent to use"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load configuration from file
        config = load_config(args.config)
        agent_config = load_agent_config(config['agent_type'])
        config['agent'].update(agent_config)
        # read COMPETITION_NAME from environment variable
        if 'COMPETITION_NAME' in os.environ:
            config['competition']['name'] = os.environ['COMPETITION_NAME']

    else:
        # Create configuration from command line arguments
        config = create_config_from_args(args)

    # Set up agent based on type
    agent_type = config['agent_type'].lower()
    execution_timeout = config['env']['execution_timeout']
    config['output_dir'] = Path(config['output_dir']) / config['competition']['name']
    config['output_dir'] = os.path.abspath(config['output_dir'])
    config['competition']['data_dir'] = Path(config['competition']['data_dir']) / config['competition']['name'] / 'data'
    config['competition']['data_dir'] = os.path.abspath(config['competition']['data_dir'])
    
    agent_handlers = {
        'mle': {
            'setup': setup_mle_agent,
            'run': run_mle_agent,
            'name': "MLE Agent"
        },
        'openai': {
            'setup': setup_openai_agent,
            'run': None,  # Special case, handled separately
            'name': "OpenAI Agent"
        },
        'aide': {
            'setup': setup_aide_agent,
            'run': run_aide_agent,
            'name': "AIDE Agent"
        },
        'dummy': {
            'setup': setup_dummy_agent,
            'run': run_dummy_agent,
            'name': "Dummy Agent"
        }
    }
    
    if agent_type not in agent_handlers:
        logger.error(f"Unknown agent type: {agent_type}. Must be one of: {', '.join(agent_handlers.keys())}")
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    handler = agent_handlers[agent_type]
    logger.info(f"Selected agent type: {handler['name']} ({agent_type})")
    
    # Special case for AIDE agent
    if agent_type == 'aide':
        agent, journal, cfg = handler['setup'](config)
        config['output_dir'] = cfg.workspace_dir
        env = setup_environment(config)
        
        # Run agent with timeout
        logger.info(f"Running with timeout of {execution_timeout} seconds")
            
        timeout_handler(
            execution_timeout,
            handler['run'],
            agent, env, journal, cfg, config
        )
    # Special case for OpenAI agent (async)
    elif agent_type == 'openai':
        env = setup_environment(config)
        agent = handler['setup'](config)
        
        # Set up history directories
        _, trajectory_file, fix_parse_error_file, cost_history_file = setup_agent_history_dir(str(config['output_dir']))
        output_files = (trajectory_file, fix_parse_error_file, cost_history_file)
        
        # Run agent asynchronously with timeout
        asyncio.run(run_openai_with_timeout(agent, env, output_files, config, execution_timeout))
    # Standard case for other agents
    else:
        env = setup_environment(config)
        agent = handler['setup'](config)
        
        # Set up history directories
        _, trajectory_file, fix_parse_error_file, cost_history_file = setup_agent_history_dir(str(config['output_dir']))
        output_files = (trajectory_file, fix_parse_error_file, cost_history_file)
        
        # Run agent with timeout
        logger.info(f"Running with timeout of {execution_timeout} seconds")
            
        timeout_handler(
            execution_timeout,
            handler['run'],
            agent, env, output_files, config
        )


if __name__ == "__main__":
    main()

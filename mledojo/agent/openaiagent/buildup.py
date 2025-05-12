"""
Setup module for Openai Agent.

This module provides functions to set up and configure the Openai Agent
for Kaggle competitions. It handles loading configurations, initializing
the agent with appropriate parameters, and preparing the agent.

The module is designed to work with the main.py script and integrates
with the MLE-Dojo framework for running AI agents on Kaggle competitions.
"""
import os
from typing import Dict, Any
from mledojo.agent.openaiagent.agent import OpenaiAgent, AgentAction
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from the project root .env file
# Calculate the project root directory (two levels up from the current file's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

def setup_openai_agent(config: Dict[str, Any]) -> OpenaiAgent:
    """
    Set up an OpenaiAgent based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OpenaiAgent instance
    """
    # Get API key from config or environment variable
    api_key = config['agent'].get('api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided in config or as OPENAI_API_KEY environment variable")
    
    # Validate model name
    model_name = config['agent']['model_name']
    valid_models = ['gpt-4.5', 'o3-mini', 'o1', 'gpt-4o-mini', 'gpt-4o']
    if model_name not in valid_models:
        raise ValueError(f"Model name must be one of {valid_models}")
    
    # For o3-mini and o1 models, we don't need to specify model settings
    model_settings = ModelSettings()
    if model_name not in ['o3-mini', 'o1']:
        model_settings = ModelSettings(
            temperature=config['agent'].get('temperature', 0.0),
            top_p=config['agent'].get('top_p', 1.0)
        )
    
    agent = OpenaiAgent(
        agent=Agent(
            name=config['agent'].get('name', "Kaggle Agent"),
            instructions=config['agent'].get('instructions', 
                'You are an excellent Kaggle grandmaster and are trying your best to solve a Kaggle competiton.'),
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=AsyncOpenAI(
                    api_key=api_key,
                    base_url=config['agent'].get('base_url', "https://api.openai.com/v1")
                )
            ),
            model_settings=model_settings,
            output_type=AgentAction
        ),
        history_length=config['agent'].get('history_length', 'all'),
        init_method=config['agent'].get('init_method', 'cold'),
        history_init=config['agent'].get('history_init', None)
    )
    
    return agent
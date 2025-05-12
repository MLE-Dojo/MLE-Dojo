"""
Setup module for MLE (Machine Learning Environment) Agent.

This module provides functions to set up and configure the MLE Agent
for Kaggle competitions. It handles loading configurations, initializing
the agent with appropriate parameters, and preparing the agent.

The module is designed to work with the main.py script and integrates
with the MLE-Dojo framework for running AI agents on Kaggle competitions.
"""
from typing import Dict, Any
from mledojo.agent.mleagent.agent import KaggleAgent, LLMConfig

def setup_mle_agent(config: Dict[str, Any]) -> KaggleAgent:
    """
    Set up a MLEAgent based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MLEAgent instance
    """
    llm_config = LLMConfig(
        model_mode=config['agent']['model_mode'],
        model_name=config['agent']['model_name'],
        port=config['agent'].get('port', 8314),
        max_completion_tokens=config['agent'].get('max_completion_tokens', 8192),
        max_prompt_tokens=config['agent'].get('max_prompt_tokens', 30000),
        temperature=config['agent'].get('temperature', 0.0),
        top_p=config['agent'].get('top_p', 1.0)
    )

    agent = KaggleAgent(
        api_idx=config['agent'].get('api_idx', 0),
        api_key=config['agent'].get('api_key', None),
        llm_config=llm_config,
        history_length=config['agent'].get('history_length', 'all'),
        init_method=config['agent'].get('init_method', 'cold'),
        history_init=config['agent'].get('history_init', None)
    )
    
    return agent
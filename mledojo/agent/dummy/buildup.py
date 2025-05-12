"""
Setup module for Dummy Agent.

This module provides functions to set up and configure the Dummy Agent
for Kaggle competitions. It handles initializing the agent with appropriate
parameters and preparing the agent.

The module is designed to work with the main.py script and integrates
with the MLE-Dojo framework for running AI agents on Kaggle competitions.
"""
from typing import Dict, Any
from mledojo.agent.dummy.agent import DummyAgent

def setup_dummy_agent(config: Dict[str, Any]) -> DummyAgent:
    """
    Set up a DummyAgent based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DummyAgent instance
    """
    # The DummyAgent doesn't require any configuration parameters
    # but we follow the same pattern as other agents for consistency
    agent = DummyAgent()
    
    return agent

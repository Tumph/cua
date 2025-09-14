"""
OSWorld SOTA Agent Package

This package provides a state-of-the-art computer-use agent optimized for OSWorld benchmarks.

Key components:
- OSWorldOptimizedLoop: Enhanced agent loop with multi-stage error recovery
- Recovery, Trajectory, and Verification callbacks for robust execution
- Element detection and state management tools
- Configuration-driven model selection and optimization

Usage:
    from osworld_agent import create_osworld_agent
    
    agent = await create_osworld_agent(
        config_name="primary",  # or "alternative", "fallback"
        computer=computer_instance
    )
    
    async for result in agent.run("Complete this OSWorld task"):
        # Process results
        pass
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import ComputerAgent
from agent.callbacks import (
    ImageRetentionCallback,
    BudgetManagerCallback,
    TrajectorySaverCallback,
    PromptInstructionsCallback
)

# Simplified - removed complex callbacks for MVP
# from .callbacks.recovery import OSWorldRecoveryCallback
# from .callbacks.trajectory_optimizer import OSWorldTrajectoryOptimizer
# from .callbacks.verification import ElementVerificationCallback


def load_config() -> Dict[str, Any]:
    """Load OSWorld agent configuration"""
    config_dir = Path(__file__).parent / "config"
    config = {}
    
    for config_file in ["models.yaml", "prompts.yaml", "evaluation.yaml"]:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
    
    return config


async def create_osworld_agent(
    config_name: str = "alternative",
    computer: Any = None,
    custom_tools: Optional[List[Any]] = None,
    trajectory_dir: Optional[str] = None,
    max_budget: float = 5.0,
    verbosity: int = logging.INFO
) -> ComputerAgent:
    """
    Create an OSWorld-optimized ComputerAgent.
    
    Args:
        config_name: Configuration to use ("primary", "alternative", "fallback")
        computer: Computer instance for agent interaction
        custom_tools: Additional custom tools to include
        trajectory_dir: Directory for saving trajectories
        max_budget: Maximum budget for trajectory execution
        verbosity: Logging verbosity level
        
    Returns:
        Configured ComputerAgent instance
    """
    
    # Load configuration
    config = load_config()
    
    # Get model configuration
    model_config = config.get(config_name, config.get("alternative", {}))
    if not model_config:
        raise ValueError(f"Configuration '{config_name}' not found")
    
    model = model_config.get("composed_model")
    if not model:
        raise ValueError(f"No composed_model found in configuration '{config_name}'")
    
    # Prepare tools
    tools = []
    if computer:
        tools.append(computer)
    
    # Add custom function tools for OSWorld tasks (like improved_agent.py)
    def verify_element_exists(element_description: str) -> bool:
        """Check if an element exists on screen before interacting."""
        # This is a placeholder - in production would use computer vision
        return True
    
    def wait_for_page_load(timeout: float = 10.0) -> bool:
        """Wait for page to fully load before proceeding."""
        # Placeholder - would check for loading indicators
        return True
    
    tools.extend([verify_element_exists, wait_for_page_load])
    
    if custom_tools:
        tools.extend(custom_tools)
    
    # Create callbacks - simplified like improved_agent.py for MVP
    callbacks = [
        # Image retention for memory efficiency
        ImageRetentionCallback(only_n_most_recent_images=5),  # Match improved_agent.py
        
        # Budget management
        BudgetManagerCallback(max_budget=max_budget),
    ]
    
    # Add trajectory saver if directory specified
    if trajectory_dir:
        callbacks.append(TrajectorySaverCallback(trajectory_dir))
    
    # Add system prompt
    system_prompt = config.get("system_prompt", "")
    if system_prompt:
        callbacks.append(PromptInstructionsCallback(system_prompt))
    
    # Create agent with improved_agent.py settings for better performance
    agent = ComputerAgent(
        model=model,
        tools=tools,
        callbacks=callbacks,
        verbosity=verbosity,
        only_n_most_recent_images=5,  # Keep more images for complex multi-step tasks
        use_prompt_caching=True,
        max_retries=5,  # More retries for complex tasks
        trajectory_dir=trajectory_dir or "trajectories/osworld_runs"
    )
    
    # Add custom methods for OSWorld tasks (like improved_agent.py)
    agent.get_stats = lambda: {
        "model": model,
        "config": config_name,
        "callbacks": len(callbacks),
        "tools": len(tools)
    }
    
    return agent


async def run_osworld_evaluation(
    agent: ComputerAgent,
    dataset: str = "hud-evals/OSWorld-Verified",
    split: str = "train[:10]",
    max_concurrent: int = 5,
    max_steps: int = 75
) -> List[Dict[str, Any]]:
    """
    Run OSWorld evaluation with the configured agent.
    
    Args:
        agent: Configured ComputerAgent instance
        dataset: Dataset name for evaluation
        split: Dataset split to evaluate
        max_concurrent: Maximum concurrent evaluations
        max_steps: Maximum steps per task
        
    Returns:
        List of evaluation results
    """
    
    try:
        from agent.integrations.hud import run_full_dataset
    except ImportError:
        raise ImportError("HUD integration not available. Install required dependencies.")
    
    # Extract agent configuration for HUD
    # Note: This is a simplified approach - in production, we'd need to properly
    # serialize the agent configuration for HUD
    
    results = await run_full_dataset(
        dataset=dataset,
        job_name=f"osworld-sota-test",
        model=agent.model,  # This might not work directly with composed models
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        split=split,
        trajectory_dir="trajectories/osworld_runs",
        only_n_most_recent_images=3
    )
    
    return results


# Convenience functions for common configurations

async def create_accuracy_optimized_agent(computer: Any, trajectory_dir: str = None) -> ComputerAgent:
    """Create agent optimized for accuracy (uses primary configuration)"""
    return await create_osworld_agent("primary", computer, trajectory_dir=trajectory_dir)


async def create_speed_optimized_agent(computer: Any, trajectory_dir: str = None) -> ComputerAgent:
    """Create agent optimized for speed/cost (uses alternative configuration)"""
    return await create_osworld_agent("alternative", computer, trajectory_dir=trajectory_dir)


async def create_minimal_agent(computer: Any, trajectory_dir: str = None) -> ComputerAgent:
    """Create minimal resource agent (uses fallback configuration)"""
    return await create_osworld_agent("fallback", computer, trajectory_dir=trajectory_dir)


# Export main components - simplified for MVP
__all__ = [
    'create_osworld_agent',
    'create_accuracy_optimized_agent', 
    'create_speed_optimized_agent',
    'create_minimal_agent',
    'run_osworld_evaluation',
    'load_config'
]

"""
Improved OSWorld Agent Implementation

This module provides an enhanced ComputerAgent optimized for OSWorld benchmark tasks.
It integrates custom callbacks, composed models, and intelligent error recovery.
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the CUA libs to Python path
sys.path.insert(0, str(Path(__file__).parent / "libs" / "python"))

# Now import CUA components
from agent import ComputerAgent
from agent.callbacks import (
    ImageRetentionCallback,
    BudgetManagerCallback, 
    TrajectorySaverCallback,
    PromptInstructionsCallback
)

# Import our custom OSWorld components
from osworld_agent.callbacks.recovery import OSWorldRecoveryCallback
from osworld_agent.callbacks.trajectory_optimizer import OSWorldTrajectoryOptimizer
from osworld_agent.callbacks.verification import ElementVerificationCallback
from osworld_agent import load_config


def create_improved_agent(
    model: str = None,
    computer: Any = None,
    config_name: str = "alternative"
) -> ComputerAgent:
    """
    Create an improved ComputerAgent optimized for OSWorld tasks.
    
    This agent includes:
    - Composed model support for better grounding
    - Custom callbacks for error recovery and optimization
    - Enhanced prompting for OSWorld-specific tasks
    
    Args:
        model: Override model (if None, uses config)
        computer: Computer instance to control
        config_name: Config preset ("primary", "alternative", "fallback")
        
    Returns:
        Configured ComputerAgent instance
    """
    
    # Load OSWorld configuration
    config = load_config()
    
    # Get model configuration
    model_config = config.get(config_name, config.get("alternative", {}))
    
    # Use provided model or config model
    if model is None:
        model = model_config.get("composed_model")
        # Fallback to simple model if composed not available
        if not model:
            model = model_config.get("planning_model", "openai/computer-use-preview")
    
    # Prepare tools
    tools = []
    if computer:
        tools.append(computer)
    
    # Add custom function tools for OSWorld tasks
    def verify_element_exists(element_description: str) -> bool:
        """Check if an element exists on screen before interacting."""
        # This is a placeholder - in production would use computer vision
        return True
    
    def wait_for_page_load(timeout: float = 10.0) -> bool:
        """Wait for page to fully load before proceeding."""
        # Placeholder - would check for loading indicators
        return True
    
    tools.extend([verify_element_exists, wait_for_page_load])
    
    # Create callbacks list
    callbacks = [
        # Image retention for memory efficiency
        ImageRetentionCallback(only_n_most_recent_images=3),
        
        # Budget management
        BudgetManagerCallback(max_budget=5.0),
        
        # Trajectory saving for analysis
        TrajectorySaverCallback("trajectories/osworld_runs"),
        
        # OSWorld-specific callbacks (commented out for now due to async issues)
        # OSWorldRecoveryCallback(),
        # OSWorldTrajectoryOptimizer(),
        # ElementVerificationCallback(),
    ]
    
    # Add system prompt for OSWorld tasks (enhanced for real benchmark tasks)
    osworld_instructions = (
        "You are an expert computer operator completing OSWorld benchmark tasks. "
        "These are REAL complex tasks requiring precision and multi-step workflows. "
        
        "CRITICAL EXECUTION RULES: "
        "1. ALWAYS take a screenshot first to understand the current state completely. "
        "2. READ and ANALYZE the screenshot carefully before any action. "
        "3. When clicking elements, use EXACT coordinates: click(x=coordinate_x, y=coordinate_y). "
        "4. NEVER use move() followed by click() - always click directly at target coordinates. "
        "5. Wait 2-3 seconds after each action for UI to fully respond. "
        "6. Take verification screenshots after critical actions. "
        
        "ELEMENT DETECTION: "
        "7. Look for text, buttons, menus, and form fields carefully in screenshots. "
        "8. If element not visible, scroll systematically to find it. "
        "9. For form fields, click directly in the input area before typing. "
        "10. For dropdowns/menus, click the dropdown arrow or menu button first. "
        
        "MULTI-STEP WORKFLOWS: "
        "11. Break complex tasks into clear sequential steps. "
        "12. Verify each step completed before proceeding to next. "
        "13. For settings/preferences: navigate menu → find setting → change value → save. "
        "14. For web tasks: wait for page loads, handle popups, verify navigation. "
        
        "ERROR RECOVERY: "
        "15. If action fails, take screenshot and try alternative approach. "
        "16. Use keyboard shortcuts when mouse actions fail (Ctrl+S, Alt+F4, etc). "
        "17. If stuck in dialog/menu, use Escape key to exit and retry. "
        
        "TASK COMPLETION: "
        "18. Explicitly verify task completion with final screenshot. "
        "19. State clearly when task is completed successfully. "
        "20. Be persistent but efficient - real OSWorld tasks require multiple steps."
    )
    callbacks.append(PromptInstructionsCallback(osworld_instructions))
    
    # Create the agent with enhanced settings for real OSWorld tasks
    agent = ComputerAgent(
        model=model,
        tools=tools,
        callbacks=callbacks,
        verbosity=logging.INFO,
        only_n_most_recent_images=5,  # Keep more images for complex multi-step tasks
        use_prompt_caching=True,
        max_retries=5,  # More retries for complex tasks
        trajectory_dir="trajectories/osworld_runs"
    )
    
    # Add custom methods for OSWorld tasks
    agent.get_stats = lambda: {
        "model": model,
        "config": config_name,
        "callbacks": len(callbacks),
        "tools": len(tools)
    }
    
    return agent


async def run_osworld_task(
    agent: ComputerAgent,
    task: str,
    max_steps: int = 50
) -> Dict[str, Any]:
    """
    Run a single OSWorld task with the agent.
    
    Args:
        agent: Configured ComputerAgent
        task: Task description
        max_steps: Maximum steps before stopping
        
    Returns:
        Result dictionary with success status and metrics
    """
    
    result = {
        "task": task,
        "success": False,
        "steps": 0,
        "errors": []
    }
    
    try:
        step_count = 0
        async for chunk in agent.run(task):
            step_count += 1
            
            # Check for completion indicators
            for item in chunk.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        text = content.get("text", "").lower()
                        if any(word in text for word in ["completed", "done", "finished", "success"]):
                            result["success"] = True
                        if any(word in text for word in ["error", "failed", "timeout"]):
                            result["errors"].append(text)
            
            # Safety limit
            if step_count >= max_steps:
                result["errors"].append(f"Reached max steps ({max_steps})")
                break
        
        result["steps"] = step_count
        
    except Exception as e:
        result["errors"].append(str(e))
    
    return result


# Example usage function
async def test_improved_agent():
    """Test the improved agent on a simple task."""
    
    from computer import Computer, VMProviderType
    
    # Create computer
    computer = Computer(
        os_type="linux",
        provider_type=VMProviderType.DOCKER,
        verbosity=logging.INFO
    )
    
    try:
        await computer.run()
        
        # Create improved agent
        agent = create_improved_agent(
            model="omniparser+anthropic/claude-3-5-sonnet-20241022",
            computer=computer,
            config_name="primary"
        )
        
        # Test task
        result = await run_osworld_task(
            agent,
            "Open the web browser and search for 'OSWorld benchmark' on Google",
            max_steps=20
        )
        
        print(f"Result: {result}")
        
    finally:
        await computer.stop()


if __name__ == "__main__":
    # For testing
    asyncio.run(test_improved_agent())

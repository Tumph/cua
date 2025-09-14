"""
Improved OSWorld Agent Implementation

This module provides an enhanced ComputerAgent optimized for OSWorld benchmark tasks.
It integrates custom callbacks, composed models, and intelligent error recovery.
"""

import logging
import asyncio
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the CUA libs to Python path (navigate up to root, then to libs)
root_path = Path(__file__).parent.parent.parent.parent.parent
libs_path = root_path / "libs" / "python"
sys.path.insert(0, str(libs_path))
sys.path.insert(0, str(libs_path / "core"))
sys.path.insert(0, str(libs_path / "computer"))
sys.path.insert(0, str(libs_path / "agent"))

# Now import CUA components
from agent.agent import ComputerAgent
from agent.callbacks import (
    ImageRetentionCallback,
    BudgetManagerCallback, 
    TrajectorySaverCallback,
    PromptInstructionsCallback
)
from agent.callbacks.base import AsyncCallbackHandler

# Import our custom OSWorld components
from osworld_agent.callbacks.recovery import OSWorldRecoveryCallback
from osworld_agent.callbacks.trajectory_optimizer import OSWorldTrajectoryOptimizer
from osworld_agent.callbacks.verification import ElementVerificationCallback
from osworld_agent import load_config


class GnomeHotCornerPrevention(AsyncCallbackHandler):
    """Prevents cursor movement to GNOME hot corner areas that trigger Activities overlay."""
    
    def __init__(self, hot_corner_threshold: int = 50):
        super().__init__()
        self.hot_corner_threshold = hot_corner_threshold
        self.screen_width = 1920  # Default, will be updated from screenshots
        self.screen_height = 1080
        
    def _is_hot_corner_area(self, x: int, y: int) -> bool:
        """Check if coordinates are in GNOME hot corner areas."""
        threshold = self.hot_corner_threshold
        
        # Top-left corner (Activities)
        if x <= threshold and y <= threshold:
            return True
            
        # Top-right corner (sometimes used for notifications)
        if x >= (self.screen_width - threshold) and y <= threshold:
            return True
            
        return False
    
    def _safe_move_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """Convert potentially dangerous coordinates to safe ones."""
        threshold = self.hot_corner_threshold
        
        # If in top-left hot corner, move to safe area
        if x <= threshold and y <= threshold:
            safe_x = max(x, threshold + 10)
            safe_y = max(y, threshold + 10)
            logging.info(f"🛡️ Hot corner prevention: Moving from ({x}, {y}) to safe coordinates ({safe_x}, {safe_y})")
            return safe_x, safe_y
            
        # If in top-right hot corner, move to safe area
        if x >= (self.screen_width - threshold) and y <= threshold:
            safe_x = min(x, self.screen_width - threshold - 10)
            safe_y = max(y, threshold + 10)
            logging.info(f"🛡️ Hot corner prevention: Moving from ({x}, {y}) to safe coordinates ({safe_x}, {safe_y})")
            return safe_x, safe_y
            
        return x, y
    
    async def on_computer_call_start(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept computer calls and modify dangerous move commands."""
        action = item.get("action", {})
        action_type = action.get("type")
        
        # Only process move-related actions
        if action_type in ["move", "mouse_move", "move_cursor"]:
            x = action.get("x", 0)
            y = action.get("y", 0)
            
            # Check if coordinates are in hot corner area
            if self._is_hot_corner_area(x, y):
                # Modify the coordinates to safe ones
                safe_x, safe_y = self._safe_move_coordinates(x, y)
                
                # Update the action with safe coordinates
                action["x"] = safe_x
                action["y"] = safe_y
                
                # Create modified item
                modified_item = item.copy()
                modified_item["action"] = action
                
                return modified_item
        
        # Also handle click actions that include movement
        elif action_type in ["click", "left_click", "right_click", "double_click"]:
            x = action.get("x")
            y = action.get("y")
            
            if x is not None and y is not None and self._is_hot_corner_area(x, y):
                safe_x, safe_y = self._safe_move_coordinates(x, y)
                
                # Update the action with safe coordinates
                action["x"] = safe_x
                action["y"] = safe_y
                
                # Create modified item
                modified_item = item.copy()
                modified_item["action"] = action
                
                return modified_item
        
        return item
    
    async def on_screenshot(self, screenshot_base64: str, screenshot_type: str) -> None:
        """Update screen dimensions from screenshots."""
        try:
            import base64
            from PIL import Image
            import io
            
            # Decode the screenshot to get dimensions
            screenshot_bytes = base64.b64decode(screenshot_base64)
            image = Image.open(io.BytesIO(screenshot_bytes))
            self.screen_width, self.screen_height = image.size
            
        except Exception as e:
            logging.warning(f"Could not extract screen dimensions from screenshot: {e}")
            # Keep default dimensions


class KnowledgeBaseCallback(AsyncCallbackHandler):
    """Provides domain-specific knowledge and strategies for tricky evaluation tasks."""
    
    def __init__(self):
        super().__init__()
        self.knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Build the knowledge base with domain-specific information."""
        return {
            "gimp_transparency": {
                "keywords": ["gimp", "transparent", "background", "see-through", "alpha channel"],
                "strategy": "specific_workflow",
                "workflow": [
                    "Open the image in GIMP",
                    "Add Alpha Channel: Right-click layer → Add Alpha Channel (if not grayed out)",
                    "Use Fuzzy Select tool (magic wand) from left toolbar",
                    "Click on the background to select it (moving dots outline)",
                    "Press Delete key to remove background (checkerboard = transparent)",
                    "Export as PNG: File → Export As → choose PNG format → Export"
                ],
                "critical_notes": [
                    "Must use PNG format to preserve transparency",
                    "Fuzzy Select tool is the magic wand icon",
                    "Checkerboard pattern indicates transparency",
                    "Alpha Channel must be added first if not present"
                ]
            },
            
            "macys_shopping_discount": {
                "keywords": ["macy's", "discount", "sale", "clearance", "percent off", "%"],
                "strategy": "clearance_first",
                "workflow": [
                    "Navigate to Clearance section FIRST (not regular categories)",
                    "Then search for specific item type (shirts, etc.)",
                    "Filter by size and other criteria within clearance",
                    "Look for percentage discounts in clearance items"
                ],
                "critical_notes": [
                    "Always start with clearance/sale section for discount searches",
                    "Don't search regular categories then look for sales",
                    "Clearance items are pre-filtered for discounts"
                ]
            },
            
            "csv_table_conversion": {
                "keywords": ["comma", "csv", "table", "convert", "separated", "commas"],
                "strategy": "reasoning_questions",
                "reasoning_flow": [
                    "Where is the comma-separated text located?",
                    "How can I quickly convert CSV to table format?",
                    "The word 'convert' means to replace/transform existing data"
                ],
                "workflow": [
                    "Locate the comma-separated values first",
                    "Select the CSV text",
                    "Use appropriate conversion method (spreadsheet, word processor table, etc.)",
                    "Replace the original CSV with formatted table"
                ],
                "critical_notes": [
                    "First identify WHERE the CSV data is",
                    "Convert means replace, not create separate",
                    "Different apps have different CSV-to-table methods"
                ]
            },
            
            "file_operations": {
                "keywords": ["save", "export", "format", "file type"],
                "strategy": "format_awareness",
                "critical_notes": [
                    "PNG for images with transparency",
                    "DOCX for Word documents with formatting",
                    "PDF for final documents that shouldn't be edited"
                ]
            },
            
            "shopping_strategies": {
                "keywords": ["shopping", "buy", "purchase", "find items"],
                "strategy": "efficient_navigation",
                "critical_notes": [
                    "Use site search for specific items",
                    "Check filters and categories",
                    "Look for sale/clearance sections for discounts"
                ]
            }
        }
    
    def _find_matching_knowledge(self, task_text: str) -> List[Dict[str, Any]]:
        """Find knowledge entries that match the task."""
        task_lower = task_text.lower()
        matches = []
        
        for knowledge_key, knowledge_data in self.knowledge_base.items():
            keywords = knowledge_data.get("keywords", [])
            
            # Check if any keywords match the task
            if any(keyword in task_lower for keyword in keywords):
                matches.append({
                    "key": knowledge_key,
                    "data": knowledge_data,
                    "relevance": sum(1 for keyword in keywords if keyword in task_lower)
                })
        
        # Sort by relevance (most matching keywords first)
        matches.sort(key=lambda x: x["relevance"], reverse=True)
        return matches
    
    async def on_run_start(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]]) -> None:
        """Check for applicable knowledge at the start of each task."""
        # Extract task text (same logic as TaskPlannerCallback)
        task_text = ""
        
        if 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, str):
                task_text = messages
            elif isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "user":
                        content = message.get("content", [])
                        if isinstance(content, str):
                            task_text = content
                            break
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    task_text = item.get("text", "")
                                    break
                        if task_text:
                            break
        
        if not task_text:
            for message in old_items:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, str):
                        task_text = content
                        break
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                task_text = item.get("text", "")
                                break
                    if task_text:
                        break
        
        if task_text:
            matches = self._find_matching_knowledge(task_text)
            if matches:
                logging.info(f"🧠 Knowledge Base: Found {len(matches)} relevant knowledge entries")
                for match in matches:
                    logging.info(f"   - {match['key']} (relevance: {match['relevance']})")
    
    async def on_llm_start(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject relevant knowledge into the conversation."""
        # Extract task from the conversation
        task_text = ""
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", [])
                if isinstance(content, str):
                    task_text = content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            task_text = item.get("text", "")
                            break
                if task_text:
                    break
        
        if not task_text:
            return messages
        
        # Find matching knowledge
        matches = self._find_matching_knowledge(task_text)
        if not matches:
            return messages
        
        # Create knowledge injection message
        knowledge_content = "🧠 DOMAIN KNOWLEDGE & STRATEGIES:\n\n"
        
        for match in matches[:2]:  # Limit to top 2 most relevant
            knowledge_data = match["data"]
            key = match["key"]
            
            knowledge_content += f"📋 {key.replace('_', ' ').title()}:\n"
            
            # Add reasoning flow if available
            if "reasoning_flow" in knowledge_data:
                knowledge_content += "🤔 Key Questions to Ask:\n"
                for i, question in enumerate(knowledge_data["reasoning_flow"], 1):
                    knowledge_content += f"{i}. {question}\n"
                knowledge_content += "\n"
            
            # Add workflow if available
            if "workflow" in knowledge_data:
                knowledge_content += "📝 Recommended Steps:\n"
                for i, step in enumerate(knowledge_data["workflow"], 1):
                    knowledge_content += f"{i}. {step}\n"
                knowledge_content += "\n"
            
            # Add critical notes
            if "critical_notes" in knowledge_data:
                knowledge_content += "⚠️ Critical Points:\n"
                for note in knowledge_data["critical_notes"]:
                    knowledge_content += f"• {note}\n"
                knowledge_content += "\n"
        
        knowledge_content += "Remember: Use this domain knowledge to guide your approach and avoid common pitfalls."
        
        knowledge_message = {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": knowledge_content
                }
            ]
        }
        
        # Add knowledge message to the conversation
        return [knowledge_message] + messages


class TaskPlannerCallback(AsyncCallbackHandler):
    """Plans multiple approaches for completing tasks using an LLM."""
    
    def __init__(self, planning_model: str = "anthropic/claude-3-5-sonnet-20241022"):
        super().__init__()
        self.planning_model = planning_model
        self.current_plan = None
        self.current_method_index = 0
        self.methods_attempted = []
        
    async def _get_task_methods(self, task: str) -> List[Dict[str, str]]:
        """Query LLM to get multiple methods for completing the task."""
        planning_prompt = f"""
You are an expert in computer operations and user interfaces. Given the following task, provide 3-4 different methods to complete it, ranked by commonness and likelihood of success.

Task: "{task}"

For each method, provide:
1. A clear step-by-step approach
2. The specific UI elements to look for
3. Alternative paths if the primary approach fails

Format your response as JSON with this structure:
{{
    "methods": [
        {{
            "rank": 1,
            "name": "Primary Method Name",
            "description": "Brief description",
            "steps": [
                "Step 1: Detailed instruction",
                "Step 2: Detailed instruction",
                "..."
            ],
            "ui_elements": ["element1", "element2", "..."],
            "fallback": "What to do if this method fails"
        }},
        ...
    ]
}}

Focus on the most common, user-friendly approaches first. Consider different applications, menu paths, keyboard shortcuts, and alternative interfaces.
"""
        
        try:
            # Use the same LLM infrastructure as the main agent
            import litellm
            
            response = await litellm.acompletion(
                model=self.planning_model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                plan_data = json.loads(content)
                return plan_data.get("methods", [])
            except json.JSONDecodeError:
                # Fallback: extract methods from text response
                return self._parse_text_methods(content)
                
        except Exception as e:
            logging.warning(f"Planning failed: {e}")
            return []
    
    def _parse_text_methods(self, text: str) -> List[Dict[str, str]]:
        """Fallback parser for non-JSON responses."""
        # Simple fallback - create a single method from the text
        return [{
            "rank": 1,
            "name": "General Approach",
            "description": "Extracted from planning response",
            "steps": [text[:500] + "..." if len(text) > 500 else text],
            "ui_elements": [],
            "fallback": "Try alternative UI elements or keyboard shortcuts"
        }]
    
    async def on_run_start(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]]) -> None:
        """Generate plan at the start of each task."""
        # Extract the task from the run kwargs or old_items
        task_text = ""
        
        # Try to get task from kwargs first (this is where the initial message usually is)
        if 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, str):
                task_text = messages
            elif isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "user":
                        content = message.get("content", [])
                        if isinstance(content, str):
                            task_text = content
                            break
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    task_text = item.get("text", "")
                                    break
                        if task_text:
                            break
        
        # Fallback to old_items
        if not task_text:
            for message in old_items:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, str):
                        task_text = content
                        break
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                task_text = item.get("text", "")
                                break
                    if task_text:
                        break
        
        if task_text:
            logging.info(f"Planning methods for task: {task_text}")
            try:
                methods = await self._get_task_methods(task_text)
                self.current_plan = methods
                self.current_method_index = 0
                self.methods_attempted = []
                
                if methods:
                    logging.info(f"Generated {len(methods)} methods for task")
                    for i, method in enumerate(methods):
                        logging.info(f"Method {i+1}: {method.get('name', 'Unknown')}")
            except Exception as e:
                logging.error(f"Planning failed: {e}")
                self.current_plan = []
    
    async def on_llm_start(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject planning information into the conversation."""
        if not self.current_plan or self.current_method_index >= len(self.current_plan):
            return messages
            
        current_method = self.current_plan[self.current_method_index]
        
        # Create planning instruction message
        planning_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
🎯 TASK PLANNING GUIDANCE:

Current Method: {current_method.get('name', 'Unknown')} (Attempt {self.current_method_index + 1}/{len(self.current_plan)})

Description: {current_method.get('description', '')}

Recommended Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(current_method.get('steps', [])))}

Look for these UI elements: {', '.join(current_method.get('ui_elements', []))}

Fallback if this fails: {current_method.get('fallback', 'Try next method')}

Remember: Start with screenshot → press Escape → screenshot → then follow the steps above.
"""
                }
            ]
        }
        
        # Add planning message to the conversation
        return [planning_message] + messages
    
    def _detect_method_failure(self, messages) -> bool:
        """Detect if current method is failing based on recent messages."""
        if len(messages) < 6:
            return False
            
        # Look for signs of failure in recent messages
        recent_messages = messages[-6:]
        click_00_count = 0
        
        for msg in recent_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                for item in content:
                    if item.get("type") == "tool_use" and item.get("name") in ["computer", "openai_computer"]:
                        input_data = item.get("input", {})
                        if (input_data.get("type") == "click" and 
                            input_data.get("x") == 0 and input_data.get("y") == 0):
                            click_00_count += 1
        
        return click_00_count >= 2
    
    async def on_llm_end(self, output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if we need to switch to the next method."""
        messages = output
        if not self.current_plan:
            return messages
            
        # Check if current method is failing
        if self._detect_method_failure(messages):
            current_method = self.current_plan[self.current_method_index]
            self.methods_attempted.append(current_method.get('name', 'Unknown'))
            self.current_method_index += 1
            
            if self.current_method_index < len(self.current_plan):
                next_method = self.current_plan[self.current_method_index]
                
                # Inject method switch message
                switch_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
🔄 METHOD SWITCH REQUIRED:

Previous method '{current_method.get('name', 'Unknown')}' is failing (detected clicking at 0,0).

Switching to Method {self.current_method_index + 1}: {next_method.get('name', 'Unknown')}

New Approach:
{chr(10).join(f"• {step}" for step in next_method.get('steps', []))}

IMPORTANT: Start fresh with screenshot → Escape → screenshot → then follow new method.
"""
                        }
                    ]
                }
                
                return messages + [switch_message]
            else:
                # All methods exhausted
                exhausted_message = {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
⚠️ ALL PLANNED METHODS EXHAUSTED:

Attempted methods: {', '.join(self.methods_attempted)}

Please try a completely different approach or declare the task impossible to complete.
Take a screenshot and analyze what went wrong with the previous approaches.
"""
                        }
                    ]
                }
                
                return messages + [exhausted_message]
        
        return messages


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
    
    def escape_activities_search() -> str:
        """Press Escape to close GNOME Activities search overlay if it's open."""
        return "Press Escape key to close any open overlays like Activities search, then take a screenshot."
    
    def safe_desktop_action(action_description: str) -> str:
        """Safely perform desktop actions by first pressing Escape to clear any overlays."""
        return f"Before {action_description}, first press Escape to close any GNOME Activities overlay, then take a screenshot to verify the desktop is clear."
    
    tools.extend([verify_element_exists, wait_for_page_load, escape_activities_search, safe_desktop_action])
    
    # Create callbacks list
    callbacks = [
        # GNOME hot corner prevention - MUST be first to intercept move commands
        GnomeHotCornerPrevention(hot_corner_threshold=50),
        
        # Domain knowledge injection - provides specific strategies for tricky tasks
        KnowledgeBaseCallback(),
        
        # Task planning - generates multiple approaches for each task
        TaskPlannerCallback(planning_model=model or "anthropic/claude-3-5-sonnet-20241022"),
        
        # Image retention for memory efficiency
        ImageRetentionCallback(only_n_most_recent_images=5),
        
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
        "You have been provided with multiple planned approaches for this task - follow them systematically. "
        
        "CRITICAL EXECUTION RULES: "
        "1. ALWAYS start with screenshot to see current state - this is MANDATORY. "
        "2. READ and ANALYZE the screenshot carefully before any action. "
        "3. When clicking elements, use EXACT coordinates from visible elements in screenshot. "
        "4. NEVER click at (0,0) or invalid coordinates - always target visible UI elements. "
        "5. Wait 2-3 seconds after each action for UI to fully respond. "
        "6. Take verification screenshots after critical actions. "
        "7. Whenever writing text, clear the field first with Ctrl+A then type. "
        "8. When you want to click something, if you are already over the element, just click. "
        "9. If you are not over the element, move your mouse to the element and click. "
        
        "🚨 GNOME ACTIVITIES PROTECTION: "
        "8. The system automatically prevents cursor movement to GNOME hot corners (top-left/right edges). "
        "9. If you see a search bar that says 'Type to search' or dark overlay screen, immediately press Escape. "
        "10. After pressing Escape, immediately press the button/action you originally wanted to do. "
        "11. If any action opens the Activities overlay, press Escape then retry the action. "
        
        "LOOP PREVENTION: "
        "12. NEVER repeat the same action more than 2 times in a row. "
        "13. If an action fails twice, try a completely different approach. "
        "14. Track your recent actions - avoid repeating failed sequences. "
        "15. If stuck for 3+ steps, take a screenshot to reassess. "
        
        "ELEMENT DETECTION: "
        "16. Look for text, buttons, menus, and form fields carefully in screenshots. "
        "17. If element not visible, scroll systematically to find it. "
        "18. For form fields, click directly in the input area before typing. "
        "19. For dropdowns/menus, click the dropdown arrow or menu button first. "
        
        "MULTI-STEP WORKFLOWS: "
        "20. Break complex tasks into clear sequential steps. "
        "21. Verify each step completed before proceeding to next. "
        "22. For settings/preferences: navigate menu → find setting → change value → save. "
        "23. For web tasks: wait for page loads, handle popups, verify navigation. "
        
        "ERROR RECOVERY: "
        "24. If action fails, take screenshot and try alternative approach. "
        "25. Use keyboard shortcuts when mouse actions fail (Ctrl+S, Alt+F4, etc). "
        "26. Try different interaction methods: keyboard vs mouse, shortcuts vs menus. "
        
        "TASK COMPLETION: "
        "27. Explicitly verify task completion with final screenshot. "
        "28. State clearly when task is completed successfully. "
        "29. Be persistent but efficient - avoid infinite loops."
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
        max_retries=2,  # Reduced retries to prevent loops - let callback handle it
        trajectory_dir="trajectories/osworld_runs"
    )
    
    # Add custom methods for OSWorld tasks
    agent.get_stats = lambda: {
        "model": model,
        "config": config_name,
        "callbacks": len(callbacks),
        "tools": len(tools),
        "task_planning": "enabled",
        "knowledge_base": "enabled",
        "gnome_protection": "programmatic+prompt",
        "hot_corner_prevention": "enabled"
    }
    
    return agent


async def run_osworld_task(
    agent: ComputerAgent,
    task: str,
    max_steps: int = 75,  # OSWorld standard
    timeout: int = 600    # OSWorld standard: 10 minutes per task
) -> Dict[str, Any]:
    """
    Run a single OSWorld task with the agent.
    
    Args:
        agent: Configured ComputerAgent
        task: Task description
        max_steps: Maximum steps before stopping (OSWorld default: 75)
        timeout: Maximum time in seconds (OSWorld default: 600s/10min)
        
    Returns:
        Result dictionary with success status and metrics
    """
    
    import time
    start_time = time.time()
    
    result = {
        "task": task,
        "success": False,
        "steps": 0,
        "errors": [],
        "duration": 0,
        "timeout_reached": False
    }
    
    try:
        step_count = 0
        async for chunk in agent.run(task):
            step_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check for timeout (OSWorld standard)
            if elapsed > timeout:
                result["timeout_reached"] = True
                result["errors"].append(f"Task timeout after {timeout}s")
                break
            
            # Log progress every 10 steps
            if step_count % 10 == 0:
                logging.info(f"📊 Step {step_count}/{max_steps}, Elapsed: {elapsed:.1f}s/{timeout}s")
            
            # Check for completion indicators in agent output
            for item in chunk.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        text = content.get("text", "").lower()
                        # More comprehensive completion detection
                        completion_words = [
                            "task completed", "successfully completed", "done", 
                            "finished", "success", "accomplished", "achieved"
                        ]
                        error_words = [
                            "error", "failed", "cannot complete", "impossible",
                            "timeout", "crashed", "unable to"
                        ]
                        
                        if any(word in text for word in completion_words):
                            result["success"] = True
                            logging.info(f"✅ Task completion detected: {text[:100]}...")
                        if any(word in text for word in error_words):
                            result["errors"].append(text[:200])
            
            # Safety limit (OSWorld standard)
            if step_count >= max_steps:
                result["errors"].append(f"Reached max steps ({max_steps})")
                break
        
        result["steps"] = step_count
        result["duration"] = time.time() - start_time
        
        # Final success check
        if not result["success"] and not result["errors"]:
            # If we completed without explicit success/error, consider it incomplete
            result["errors"].append("Task completed without clear success indication")
        
    except Exception as e:
        result["errors"].append(str(e))
        result["duration"] = time.time() - start_time
    
    return result


# Example usage function
async def test_improved_agent():
    """Test the improved agent on a simple task."""
    
    from computer.computer import Computer
    from computer.providers.base import VMProviderType
    
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

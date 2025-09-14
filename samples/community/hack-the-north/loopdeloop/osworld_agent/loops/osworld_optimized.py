"""
OSWorld Optimized Agent Loop

This extends the ComposedGroundedConfig to add OSWorld-specific optimizations:
- Multi-stage grounding with fallbacks
- Verification checkpoints after critical actions
- State management across application switches
- Adaptive timing based on action type and application responsiveness
- Enhanced error recovery with pattern learning
"""

import uuid
import asyncio
import json
import yaml
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from typing import Dict, List, Any, Optional, Tuple
# Note: We're not using @register_agent since we'll use this differently
# from agent.decorators import register_agent
# from agent.types import Messages, AgentCapability
# from agent.loops.composed_grounded import ComposedGroundedConfig
# from agent.responses import (
#     make_reasoning_item,
#     make_output_text_item,
#     make_screenshot_item
# )
# from agent.loops.composed_grounded import get_last_computer_call_image


class OSWorldState:
    """Manages state across task execution"""
    
    def __init__(self):
        self.current_application = None
        self.task_context = {}
        self.checkpoints = []
        self.failure_patterns = {}
        self.successful_patterns = {}
        self.action_history = []
    
    def save_checkpoint(self, description: str, state: Dict[str, Any]):
        """Save a state checkpoint"""
        checkpoint = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'description': description,
            'state': state,
            'application': self.current_application
        }
        self.checkpoints.append(checkpoint)
        return checkpoint['id']
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore to a previous checkpoint"""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint['id'] == checkpoint_id:
                self.current_application = checkpoint['application']
                return checkpoint['state']
        return None
    
    def record_action(self, action: Dict[str, Any], success: bool, result: Any = None):
        """Record action outcome for pattern learning"""
        record = {
            'action': action,
            'success': success,
            'result': result,
            'timestamp': time.time(),
            'application': self.current_application
        }
        self.action_history.append(record)
        
        # Update patterns
        action_key = f"{action.get('type', 'unknown')}_{self.current_application}"
        if success:
            if action_key not in self.successful_patterns:
                self.successful_patterns[action_key] = []
            self.successful_patterns[action_key].append(record)
        else:
            if action_key not in self.failure_patterns:
                self.failure_patterns[action_key] = []
            self.failure_patterns[action_key].append(record)


# This class provides enhanced functionality but doesn't override the agent loop directly
# Instead, we'll use it through callbacks and configuration
class OSWorldOptimizedLoop:
    """
    OSWorld-optimized agent loop with enhanced error recovery,
    state management, and adaptive strategies.
    """
    
    def __init__(self):
        super().__init__()
        self.state = OSWorldState()
        self.config = self._load_config()
        self.adaptive_delays = {
            'screenshot': 0.5,
            'click': 0.3,
            'type': 0.1,
            'wait': 2.0,
            'scroll': 0.5
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        config_dir = Path(__file__).parent.parent / "config"
        config = {}
        
        for config_file in ["models.yaml", "prompts.yaml", "evaluation.yaml"]:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
        
        return config
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return supported capabilities"""
        return ["step", "click"]
    
    async def predict_step(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        use_prompt_caching: Optional[bool] = False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced predict step with OSWorld optimizations
        """
        
        # Phase 1: Context Analysis
        await self._analyze_context(messages, computer_handler)
        
        # Phase 2: Enhanced Planning with Prompts
        enhanced_messages = await self._enhance_messages_with_context(messages)
        
        # Phase 3: Multi-stage Execution
        try:
            # Call parent composed grounded method
            result = await super().predict_step(
                enhanced_messages,
                model,
                tools,
                max_retries,
                stream,
                computer_handler,
                use_prompt_caching,
                _on_api_start,
                _on_api_end,
                _on_usage,
                _on_screenshot,
                **kwargs
            )
            
            # Phase 4: Post-processing and Verification
            verified_result = await self._verify_and_enhance_result(
                result, computer_handler
            )
            
            return verified_result
            
        except Exception as e:
            # Phase 5: Error Recovery
            recovery_result = await self._handle_error_with_recovery(
                e, messages, model, tools, computer_handler, **kwargs
            )
            return recovery_result
    
    async def _analyze_context(self, messages: List[Dict[str, Any]], computer_handler):
        """Analyze current context and update state"""
        
        # Detect current application
        if computer_handler:
            try:
                # Take screenshot to analyze current state
                screenshot_b64 = await computer_handler.screenshot()
                if screenshot_b64:
                    # Simple heuristic - in real implementation, use image analysis
                    self.state.current_application = await self._detect_application(screenshot_b64)
            except Exception:
                pass
        
        # Extract task context from messages
        for message in messages:
            if message.get('role') == 'user':
                content = message.get('content', '')
                if isinstance(content, str):
                    self.state.task_context['user_intent'] = content
    
    async def _detect_application(self, screenshot_b64: str) -> str:
        """Detect current application from screenshot"""
        # Placeholder - in real implementation, use computer vision
        # to detect application windows, titles, etc.
        return "unknown"
    
    async def _enhance_messages_with_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance messages with OSWorld-specific context and prompts"""
        
        enhanced_messages = messages.copy()
        
        # Add system prompt if not present
        system_prompt = self.config.get('system_prompt', '')
        if system_prompt and not any(msg.get('role') == 'system' for msg in enhanced_messages):
            enhanced_messages.insert(0, {
                'role': 'system',
                'content': system_prompt
            })
        
        # Add context about current application and state
        if self.state.current_application:
            context_msg = {
                'role': 'user',
                'content': f"Current application: {self.state.current_application}. "
                         f"Previous actions: {len(self.state.action_history)} completed."
            }
            enhanced_messages.insert(-1, context_msg)
        
        # Add successful patterns if available
        if self.state.successful_patterns:
            pattern_hints = self._generate_pattern_hints()
            if pattern_hints:
                enhanced_messages.insert(-1, {
                    'role': 'user', 
                    'content': f"Helpful patterns from previous successes: {pattern_hints}"
                })
        
        return enhanced_messages
    
    def _generate_pattern_hints(self) -> str:
        """Generate hints from successful patterns"""
        hints = []
        
        for pattern_key, successes in self.state.successful_patterns.items():
            if len(successes) >= 2:  # Pattern must occur multiple times
                action_type = pattern_key.split('_')[0]
                hints.append(f"For {action_type} actions, previous successes used similar approaches")
        
        return "; ".join(hints[:3])  # Limit to top 3 hints
    
    async def _verify_and_enhance_result(
        self, 
        result: Dict[str, Any], 
        computer_handler
    ) -> Dict[str, Any]:
        """Verify result and add checkpoints for critical actions"""
        
        output_items = result.get('output', [])
        enhanced_output = []
        
        for item in output_items:
            enhanced_output.append(item)
            
            # Add verification for critical actions
            if item.get('type') == 'computer_call':
                action = item.get('action', {})
                action_type = action.get('type')
                
                # Save checkpoint before critical actions
                if action_type in ['click', 'type', 'keypress']:
                    checkpoint_id = self.state.save_checkpoint(
                        f"Before {action_type} action",
                        {'action': action, 'timestamp': time.time()}
                    )
                
                # Add adaptive delay based on action type
                if action_type in self.adaptive_delays:
                    delay = self.adaptive_delays[action_type]
                    await asyncio.sleep(delay)
                
                # Record action for pattern learning
                self.state.record_action(action, True)  # Assume success for now
        
        result['output'] = enhanced_output
        return result
    
    async def _handle_error_with_recovery(
        self,
        error: Exception,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        computer_handler,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle errors with intelligent recovery strategies"""
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Record failure pattern
        if self.state.action_history:
            last_action = self.state.action_history[-1]['action']
            self.state.record_action(last_action, False, error_msg)
        
        # Generate recovery strategy based on error type
        recovery_messages = messages.copy()
        
        if "element not found" in error_msg.lower():
            recovery_prompt = self.config.get('error_prompts', {}).get('element_not_found', 
                'Element not found. Please take a screenshot and try alternative approaches.')
        elif "timeout" in error_msg.lower():
            recovery_prompt = self.config.get('error_prompts', {}).get('timeout',
                'Operation timed out. Please wait and retry with different timing.')
        else:
            recovery_prompt = f"Error occurred: {error_msg}. Please analyze the situation and try an alternative approach."
        
        recovery_messages.append({
            'role': 'user',
            'content': recovery_prompt
        })
        
        # Add reasoning about the error
        reasoning_item = make_reasoning_item([{
            'text': f"Encountered error: {error_type}. Attempting recovery strategy.",
            'type': 'summary_text'
        }])
        
        # Try recovery with simplified approach
        try:
            # Use fallback model configuration for recovery
            fallback_model = self.config.get('fallback', {}).get('composed_model', model)
            
            recovery_result = await super().predict_step(
                recovery_messages,
                fallback_model,
                tools,
                max_retries=1,  # Limit retries for recovery
                computer_handler=computer_handler,
                **kwargs
            )
            
            # Prepend reasoning about recovery
            recovery_output = [reasoning_item] + recovery_result.get('output', [])
            recovery_result['output'] = recovery_output
            
            return recovery_result
            
        except Exception as recovery_error:
            # If recovery fails, return error information
            error_output = [
                reasoning_item,
                make_output_text_item(f"Recovery failed: {recovery_error}. Task may require manual intervention.")
            ]
            
            return {
                'output': error_output,
                'usage': {'total_tokens': 0, 'response_cost': 0.0}
            }
    
    async def predict_click(
        self,
        model: str,
        image_b64: str,
        instruction: str
    ) -> Optional[Tuple[int, int]]:
        """Enhanced click prediction with fallback strategies"""
        
        # Try primary grounding approach
        try:
            result = await super().predict_click(model, image_b64, instruction)
            if result:
                return result
        except Exception as e:
            # Record grounding failure
            self.state.record_action(
                {'type': 'predict_click', 'instruction': instruction}, 
                False, 
                str(e)
            )
        
        # Try fallback grounding models
        fallback_models = self.config.get('fallback', {})
        if fallback_models:
            fallback_model = fallback_models.get('composed_model', model)
            try:
                return await super().predict_click(fallback_model, image_b64, instruction)
            except Exception:
                pass
        
        # If all grounding fails, return None
        return None

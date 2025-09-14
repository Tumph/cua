"""
Element Verification Callback

Pre-validates element visibility/interactability and post-validates action success:
- Element existence verification before actions
- Action success validation after execution
- Dynamic UI change detection
- Intelligent waiting strategies
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from agent.callbacks.base import AsyncCallbackHandler


class ElementVerificationCallback(AsyncCallbackHandler):
    """
    Verifies elements before interaction and validates action success.
    
    Implements intelligent verification strategies to improve reliability
    of computer actions in dynamic UI environments.
    """
    
    def __init__(self, verification_timeout: float = 10.0, max_wait_attempts: int = 5):
        self.verification_timeout = verification_timeout
        self.max_wait_attempts = max_wait_attempts
        self.verification_cache = {}
        self.last_screenshot_hash = None
    
    async def on_computer_call_start(self, call_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pre-validate element before action execution"""
        
        action = call_item.get('action', {})
        action_type = action.get('type')
        
        # Skip verification for actions that don't need elements
        if action_type in ['screenshot', 'wait', 'get_current_url', 'get_dimensions']:
            return None
        
        # Verify element for interactive actions
        if action_type in ['click', 'double_click', 'type', 'scroll', 'move']:
            verification_result = await self._verify_element_exists(action)
            
            if not verification_result['exists']:
                # Element not found - suggest waiting or alternative approach
                return {
                    'type': 'verification_failure',
                    'reason': 'element_not_found',
                    'suggestion': verification_result['suggestion'],
                    'action': action
                }
        
        return None
    
    async def on_computer_call_end(self, call_item: Dict[str, Any], result: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Post-validate action success"""
        
        action = call_item.get('action', {})
        action_type = action.get('type')
        
        # Skip validation for non-interactive actions
        if action_type in ['screenshot', 'wait', 'get_current_url', 'get_dimensions']:
            return None
        
        # Validate action success
        validation_result = await self._validate_action_success(action, result)
        
        if not validation_result['success']:
            # Action may have failed - add recovery suggestions
            recovery_actions = await self._generate_recovery_actions(action, validation_result)
            
            if recovery_actions:
                return result + recovery_actions
        
        return None
    
    async def _verify_element_exists(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that target element exists and is interactable"""
        
        element_description = action.get('element_description')
        coordinates = (action.get('x'), action.get('y'))
        
        verification_result = {
            'exists': False,
            'interactable': False,
            'suggestion': None,
            'confidence': 0.0
        }
        
        # If we have element description, try to verify it exists
        if element_description:
            # In a full implementation, this would use computer vision
            # to verify the element exists on screen
            verification_result.update(await self._check_element_by_description(element_description))
        
        # If we have coordinates, verify they're clickable
        elif coordinates and coordinates[0] is not None and coordinates[1] is not None:
            verification_result.update(await self._check_coordinates_clickable(coordinates))
        
        # Generate suggestions if element not found
        if not verification_result['exists']:
            verification_result['suggestion'] = await self._generate_element_search_suggestion(action)
        
        return verification_result
    
    async def _check_element_by_description(self, description: str) -> Dict[str, Any]:
        """Check if element exists by description"""
        
        # Placeholder implementation
        # In a real system, this would:
        # 1. Take a screenshot
        # 2. Use OCR/computer vision to find elements matching description
        # 3. Check if element is visible and clickable
        
        # For now, assume elements exist with some confidence based on description
        confidence = 0.8 if any(keyword in description.lower() for keyword in ['button', 'link', 'input']) else 0.5
        
        return {
            'exists': confidence > 0.6,
            'interactable': confidence > 0.7,
            'confidence': confidence
        }
    
    async def _check_coordinates_clickable(self, coordinates: tuple) -> Dict[str, Any]:
        """Check if coordinates are clickable"""
        
        x, y = coordinates
        
        # Basic bounds checking
        # In a real implementation, this would check:
        # 1. Coordinates are within screen bounds
        # 2. No overlapping elements blocking the click
        # 3. Element at coordinates is interactive
        
        valid_bounds = 0 <= x <= 2000 and 0 <= y <= 1500  # Reasonable screen bounds
        
        return {
            'exists': valid_bounds,
            'interactable': valid_bounds,
            'confidence': 0.9 if valid_bounds else 0.1
        }
    
    async def _generate_element_search_suggestion(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggestions for finding missing elements"""
        
        action_type = action.get('type')
        element_description = action.get('element_description', '')
        
        suggestions = {
            'action': 'wait_and_retry',
            'reason': 'Element may still be loading',
            'wait_time': 2.0,
            'alternatives': []
        }
        
        # Suggest scrolling for buttons and links
        if any(keyword in element_description.lower() for keyword in ['button', 'link', 'menu']):
            suggestions['alternatives'].append({
                'action': 'scroll_to_find',
                'direction': 'down',
                'reason': 'Element might be below current view'
            })
        
        # Suggest waiting for form elements
        if any(keyword in element_description.lower() for keyword in ['input', 'field', 'textbox']):
            suggestions['wait_time'] = 3.0
            suggestions['reason'] = 'Form elements may take time to load'
        
        # Suggest checking for dialogs/modals
        if 'not found' in element_description.lower():
            suggestions['alternatives'].append({
                'action': 'check_for_dialogs',
                'reason': 'Modal dialog might be blocking interaction'
            })
        
        return suggestions
    
    async def _validate_action_success(self, action: Dict[str, Any], result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that action was successful"""
        
        action_type = action.get('type')
        
        validation_result = {
            'success': True,
            'confidence': 0.8,
            'indicators': [],
            'issues': []
        }
        
        # Check for error indicators in result
        for item in result:
            if item.get('type') == 'error':
                validation_result['success'] = False
                validation_result['issues'].append('Error in result')
                validation_result['confidence'] = 0.0
        
        # Action-specific validation
        if action_type == 'click':
            validation_result.update(await self._validate_click_success(action, result))
        elif action_type == 'type':
            validation_result.update(await self._validate_type_success(action, result))
        elif action_type == 'scroll':
            validation_result.update(await self._validate_scroll_success(action, result))
        
        return validation_result
    
    async def _validate_click_success(self, action: Dict[str, Any], result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate click action success"""
        
        # Look for visual changes in screenshot
        screenshot_changed = await self._detect_screenshot_change(result)
        
        validation = {
            'success': True,
            'confidence': 0.7,
            'indicators': [],
            'issues': []
        }
        
        if screenshot_changed:
            validation['indicators'].append('Screenshot changed after click')
            validation['confidence'] = 0.9
        else:
            validation['confidence'] = 0.5
            validation['issues'].append('No visible change detected after click')
        
        return validation
    
    async def _validate_type_success(self, action: Dict[str, Any], result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate type action success"""
        
        text_to_type = action.get('text', '')
        
        validation = {
            'success': True,
            'confidence': 0.8,
            'indicators': [f'Typed text: "{text_to_type}"'],
            'issues': []
        }
        
        # In a full implementation, this would:
        # 1. Check if text appears in the focused element
        # 2. Verify cursor position changed
        # 3. Look for text validation errors
        
        return validation
    
    async def _validate_scroll_success(self, action: Dict[str, Any], result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate scroll action success"""
        
        scroll_x = action.get('scroll_x', 0)
        scroll_y = action.get('scroll_y', 0)
        
        validation = {
            'success': True,
            'confidence': 0.8,
            'indicators': [f'Scrolled by ({scroll_x}, {scroll_y})'],
            'issues': []
        }
        
        # Check for visual changes indicating scroll
        screenshot_changed = await self._detect_screenshot_change(result)
        
        if screenshot_changed:
            validation['confidence'] = 0.9
        else:
            validation['confidence'] = 0.6
            validation['issues'].append('No visible scroll change detected')
        
        return validation
    
    async def _detect_screenshot_change(self, result: List[Dict[str, Any]]) -> bool:
        """Detect if screenshot changed compared to previous"""
        
        # Look for screenshot in result
        for item in result:
            if (item.get('type') == 'computer_call_output' and 
                item.get('output', {}).get('type') == 'input_image'):
                
                image_url = item['output'].get('image_url', '')
                if image_url.startswith('data:image/png;base64,'):
                    current_hash = hash(image_url)
                    
                    if self.last_screenshot_hash is None:
                        self.last_screenshot_hash = current_hash
                        return True
                    
                    changed = current_hash != self.last_screenshot_hash
                    self.last_screenshot_hash = current_hash
                    return changed
        
        return False
    
    async def _generate_recovery_actions(self, action: Dict[str, Any], validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recovery actions for failed validations"""
        
        recovery_actions = []
        
        # If action seems to have failed, suggest alternatives
        if not validation_result['success'] or validation_result['confidence'] < 0.5:
            
            action_type = action.get('type')
            
            # For clicks that didn't seem to work
            if action_type == 'click':
                # Try double-click
                recovery_actions.append({
                    'type': 'computer_call',
                    'action': {
                        'type': 'double_click',
                        **{k: v for k, v in action.items() if k != 'type'}
                    },
                    'call_id': f"recovery_double_click_{int(time.time())}"
                })
                
                # Try right-click to see context menu
                recovery_actions.append({
                    'type': 'computer_call',
                    'action': {
                        'type': 'click',
                        'button': 'right',
                        **{k: v for k, v in action.items() if k not in ['type', 'button']}
                    },
                    'call_id': f"recovery_right_click_{int(time.time())}"
                })
            
            # For typing that may not have worked
            elif action_type == 'type':
                # Clear field first, then retype
                recovery_actions.extend([
                    {
                        'type': 'computer_call',
                        'action': {
                            'type': 'keypress',
                            'keys': ['ctrl+a']  # Select all
                        },
                        'call_id': f"recovery_select_all_{int(time.time())}"
                    },
                    {
                        'type': 'computer_call',
                        'action': action.copy(),  # Retype original text
                        'call_id': f"recovery_retype_{int(time.time())}"
                    }
                ])
            
            # Add a wait action to let UI settle
            recovery_actions.append({
                'type': 'computer_call',
                'action': {'type': 'wait'},
                'call_id': f"recovery_wait_{int(time.time())}"
            })
        
        return recovery_actions
    
    async def wait_for_element(self, element_description: str, timeout: float = None) -> bool:
        """Wait for element to appear with timeout"""
        
        timeout = timeout or self.verification_timeout
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout and attempt < self.max_wait_attempts:
            # Check if element exists now
            verification = await self._check_element_by_description(element_description)
            
            if verification['exists']:
                return True
            
            # Wait before next attempt
            wait_time = min(2.0 ** attempt, 5.0)  # Exponential backoff, max 5s
            await asyncio.sleep(wait_time)
            attempt += 1
        
        return False
    
    async def wait_for_condition(self, condition_func, timeout: float = None) -> bool:
        """Wait for custom condition to be met"""
        
        timeout = timeout or self.verification_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await condition_func():
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(0.5)
        
        return False
    
    def clear_cache(self):
        """Clear verification cache"""
        self.verification_cache.clear()
        self.last_screenshot_hash = None
    
    async def get_verification_statistics(self) -> Dict[str, Any]:
        """Get verification statistics"""
        return {
            'cache_size': len(self.verification_cache),
            'timeout_setting': self.verification_timeout,
            'max_wait_attempts': self.max_wait_attempts,
            'last_screenshot_cached': self.last_screenshot_hash is not None
        }

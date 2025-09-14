"""
OSWorld Recovery Callback

Implements intelligent error recovery strategies for OSWorld tasks:
- Pattern-based failure analysis
- Multi-stage recovery approaches
- Failure database maintenance
- Adaptive strategy selection
"""

import json
import time
import sqlite3
from typing import Any, Dict, List, Optional
from pathlib import Path

from agent.callbacks.base import AsyncCallbackHandler


class OSWorldRecoveryCallback(AsyncCallbackHandler):
    """
    Advanced error recovery system for OSWorld tasks.
    
    Maintains a database of failure patterns and implements
    intelligent recovery strategies based on historical data.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "trajectories/failure_patterns.db"
        self.recovery_strategies = {
            'element_not_found': self._recover_element_not_found,
            'timeout': self._recover_timeout,
            'permission_denied': self._recover_permission_denied,
            'network_error': self._recover_network_error,
            'application_error': self._recover_application_error
        }
        self._init_database()
    
    def _init_database(self):
        """Initialize failure pattern database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                context TEXT NOT NULL,
                recovery_strategy TEXT,
                success_rate REAL DEFAULT 0.0,
                attempt_count INTEGER DEFAULT 0,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recovery_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                strategy TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                FOREIGN KEY (pattern_id) REFERENCES failure_patterns (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def on_computer_call_error(self, error: Exception, call_item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Handle computer call errors with recovery strategies"""
        
        error_type = self._classify_error(error)
        error_message = str(error)
        context = self._extract_context(call_item)
        
        # Record the failure
        pattern_id = self._record_failure(error_type, error_message, context)
        
        # Get best recovery strategy
        strategy = await self._get_recovery_strategy(error_type, error_message, context)
        
        if strategy:
            # Attempt recovery
            recovery_actions = await strategy(error, call_item, context)
            
            # Record recovery attempt
            success = recovery_actions is not None and len(recovery_actions) > 0
            self._record_recovery_attempt(pattern_id, strategy.__name__, success, recovery_actions)
            
            return recovery_actions
        
        return None
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery strategy selection"""
        error_msg = str(error).lower()
        
        if "element not found" in error_msg or "selector" in error_msg:
            return "element_not_found"
        elif "timeout" in error_msg or "timed out" in error_msg:
            return "timeout"
        elif "permission" in error_msg or "access denied" in error_msg:
            return "permission_denied"
        elif "network" in error_msg or "connection" in error_msg:
            return "network_error"
        elif "application" in error_msg or "window" in error_msg:
            return "application_error"
        else:
            return "unknown"
    
    def _extract_context(self, call_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from the failed call"""
        action = call_item.get('action', {})
        return {
            'action_type': action.get('type'),
            'element_description': action.get('element_description'),
            'coordinates': (action.get('x'), action.get('y')),
            'text': action.get('text'),
            'call_id': call_item.get('call_id')
        }
    
    def _record_failure(self, error_type: str, error_message: str, context: Dict[str, Any]) -> int:
        """Record failure pattern in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        context_json = json.dumps(context)
        
        # Check if this pattern exists
        cursor.execute('''
            SELECT id, attempt_count FROM failure_patterns 
            WHERE error_type = ? AND error_message = ? AND context = ?
        ''', (error_type, error_message, context_json))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing pattern
            pattern_id, attempt_count = result
            cursor.execute('''
                UPDATE failure_patterns 
                SET attempt_count = ?, last_seen = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (attempt_count + 1, pattern_id))
        else:
            # Create new pattern
            cursor.execute('''
                INSERT INTO failure_patterns (error_type, error_message, context, attempt_count)
                VALUES (?, ?, ?, 1)
            ''', (error_type, error_message, context_json))
            pattern_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return pattern_id
    
    def _record_recovery_attempt(self, pattern_id: int, strategy: str, success: bool, details: Any):
        """Record recovery attempt outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        details_json = json.dumps(details) if details else None
        
        cursor.execute('''
            INSERT INTO recovery_attempts (pattern_id, strategy, success, details)
            VALUES (?, ?, ?, ?)
        ''', (pattern_id, strategy, success, details_json))
        
        # Update success rate for the pattern
        cursor.execute('''
            SELECT COUNT(*) as total, SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
            FROM recovery_attempts WHERE pattern_id = ?
        ''', (pattern_id,))
        
        total, successes = cursor.fetchone()
        success_rate = (successes or 0) / total if total > 0 else 0.0
        
        cursor.execute('''
            UPDATE failure_patterns SET success_rate = ? WHERE id = ?
        ''', (success_rate, pattern_id))
        
        conn.commit()
        conn.close()
    
    async def _get_recovery_strategy(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Select best recovery strategy based on historical data"""
        
        # First, try to find the best strategy from history
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ra.strategy, AVG(CASE WHEN ra.success THEN 1.0 ELSE 0.0 END) as success_rate
            FROM recovery_attempts ra
            JOIN failure_patterns fp ON ra.pattern_id = fp.id
            WHERE fp.error_type = ?
            GROUP BY ra.strategy
            ORDER BY success_rate DESC, COUNT(*) DESC
            LIMIT 1
        ''', (error_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[1] > 0.3:  # Only use if success rate > 30%
            strategy_name = result[0]
            return self.recovery_strategies.get(strategy_name.replace('_recover_', ''))
        
        # Fall back to default strategy for error type
        return self.recovery_strategies.get(error_type)
    
    async def _recover_element_not_found(self, error: Exception, call_item: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recovery strategy for element not found errors"""
        action = call_item.get('action', {})
        element_desc = action.get('element_description', '')
        
        recovery_actions = []
        
        # Strategy 1: Take screenshot to see current state
        recovery_actions.append({
            'type': 'computer_call',
            'action': {'type': 'screenshot'},
            'call_id': f"recovery_screenshot_{int(time.time())}"
        })
        
        # Strategy 2: Try scrolling to find element
        if 'button' in element_desc.lower() or 'link' in element_desc.lower():
            recovery_actions.append({
                'type': 'computer_call',
                'action': {
                    'type': 'scroll',
                    'x': 500,  # Center of typical screen
                    'y': 400,
                    'scroll_x': 0,
                    'scroll_y': -3  # Scroll up
                },
                'call_id': f"recovery_scroll_{int(time.time())}"
            })
        
        # Strategy 3: Try alternative element descriptions
        alternative_descriptions = self._generate_alternative_descriptions(element_desc)
        for alt_desc in alternative_descriptions[:2]:  # Limit to 2 alternatives
            recovery_actions.append({
                'type': 'computer_call',
                'action': {
                    'type': action.get('type', 'click'),
                    'element_description': alt_desc,
                    **{k: v for k, v in action.items() if k not in ['type', 'element_description']}
                },
                'call_id': f"recovery_alt_{int(time.time())}"
            })
        
        return recovery_actions
    
    async def _recover_timeout(self, error: Exception, call_item: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recovery strategy for timeout errors"""
        
        recovery_actions = []
        
        # Strategy 1: Wait longer
        recovery_actions.append({
            'type': 'computer_call',
            'action': {'type': 'wait'},
            'call_id': f"recovery_wait_{int(time.time())}"
        })
        
        # Strategy 2: Take screenshot to see if anything changed
        recovery_actions.append({
            'type': 'computer_call',
            'action': {'type': 'screenshot'},
            'call_id': f"recovery_screenshot_{int(time.time())}"
        })
        
        # Strategy 3: Retry original action with longer timeout
        original_action = call_item.get('action', {}).copy()
        recovery_actions.append({
            'type': 'computer_call',
            'action': original_action,
            'call_id': f"recovery_retry_{int(time.time())}"
        })
        
        return recovery_actions
    
    async def _recover_permission_denied(self, error: Exception, call_item: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recovery strategy for permission denied errors"""
        
        # For permission errors, we need to be careful
        # Usually requires user intervention or alternative approach
        return [{
            'type': 'message',
            'role': 'assistant',
            'content': [{
                'type': 'output_text',
                'text': f"Permission denied error encountered. This may require administrator privileges or an alternative approach. Error: {error}"
            }]
        }]
    
    async def _recover_network_error(self, error: Exception, call_item: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recovery strategy for network errors"""
        
        recovery_actions = []
        
        # Strategy 1: Wait for network recovery
        recovery_actions.append({
            'type': 'computer_call',
            'action': {'type': 'wait'},
            'call_id': f"recovery_network_wait_{int(time.time())}"
        })
        
        # Strategy 2: Refresh page if it's a web action
        action = call_item.get('action', {})
        if action.get('type') in ['click', 'type'] and 'browser' in str(context).lower():
            recovery_actions.append({
                'type': 'computer_call',
                'action': {
                    'type': 'keypress',
                    'keys': ['F5']  # Refresh page
                },
                'call_id': f"recovery_refresh_{int(time.time())}"
            })
        
        return recovery_actions
    
    async def _recover_application_error(self, error: Exception, call_item: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recovery strategy for application-specific errors"""
        
        recovery_actions = []
        
        # Strategy 1: Take screenshot to see error dialogs
        recovery_actions.append({
            'type': 'computer_call',
            'action': {'type': 'screenshot'},
            'call_id': f"recovery_app_screenshot_{int(time.time())}"
        })
        
        # Strategy 2: Try to close error dialogs
        recovery_actions.append({
            'type': 'computer_call',
            'action': {
                'type': 'keypress',
                'keys': ['Escape']
            },
            'call_id': f"recovery_escape_{int(time.time())}"
        })
        
        return recovery_actions
    
    def _generate_alternative_descriptions(self, original_desc: str) -> List[str]:
        """Generate alternative element descriptions"""
        alternatives = []
        
        # Simple word substitutions
        substitutions = {
            'button': ['btn', 'link', 'clickable element'],
            'click': ['select', 'choose', 'press'],
            'submit': ['send', 'confirm', 'ok'],
            'close': ['dismiss', 'cancel', 'x'],
            'menu': ['dropdown', 'options', 'list']
        }
        
        desc_lower = original_desc.lower()
        for original, alternatives_list in substitutions.items():
            if original in desc_lower:
                for alt in alternatives_list:
                    new_desc = desc_lower.replace(original, alt)
                    alternatives.append(new_desc)
        
        # Add more generic descriptions
        if 'button' in desc_lower:
            alternatives.append('clickable element')
        if any(word in desc_lower for word in ['red', 'green', 'blue']):
            alternatives.append('colored button')
        
        return alternatives[:3]  # Limit alternatives
    
    async def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute('''
            SELECT 
                error_type,
                COUNT(*) as total_failures,
                AVG(success_rate) as avg_success_rate,
                SUM(attempt_count) as total_attempts
            FROM failure_patterns
            GROUP BY error_type
            ORDER BY total_failures DESC
        ''')
        
        stats = {
            'error_types': [],
            'total_patterns': 0,
            'overall_recovery_rate': 0.0
        }
        
        total_patterns = 0
        weighted_success = 0.0
        
        for row in cursor.fetchall():
            error_type, total_failures, avg_success_rate, total_attempts = row
            stats['error_types'].append({
                'type': error_type,
                'total_failures': total_failures,
                'success_rate': avg_success_rate or 0.0,
                'total_attempts': total_attempts
            })
            
            total_patterns += total_failures
            weighted_success += (avg_success_rate or 0.0) * total_failures
        
        stats['total_patterns'] = total_patterns
        if total_patterns > 0:
            stats['overall_recovery_rate'] = weighted_success / total_patterns
        
        conn.close()
        return stats

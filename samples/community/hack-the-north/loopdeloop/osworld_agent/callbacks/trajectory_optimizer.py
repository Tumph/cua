"""
OSWorld Trajectory Optimizer Callback

Learns from successful trajectories and suggests optimized action sequences:
- Pattern extraction from successful runs
- Action sequence optimization
- Context-aware suggestions
- Performance improvement tracking
"""

import json
import sqlite3
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from agent.callbacks.base import AsyncCallbackHandler


class TrajectoryPattern:
    """Represents a learned trajectory pattern"""
    
    def __init__(self, pattern_id: str, actions: List[Dict], context: Dict, success_rate: float):
        self.pattern_id = pattern_id
        self.actions = actions
        self.context = context
        self.success_rate = success_rate
        self.usage_count = 0
        self.last_used = None
    
    def matches_context(self, current_context: Dict, threshold: float = 0.7) -> bool:
        """Check if current context matches this pattern's context"""
        if not self.context or not current_context:
            return False
        
        matches = 0
        total = 0
        
        for key in ['application', 'task_type', 'element_type']:
            if key in self.context and key in current_context:
                total += 1
                if self.context[key] == current_context[key]:
                    matches += 1
        
        return (matches / total) >= threshold if total > 0 else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage"""
        return {
            'pattern_id': self.pattern_id,
            'actions': self.actions,
            'context': self.context,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'last_used': self.last_used
        }


class OSWorldTrajectoryOptimizer(AsyncCallbackHandler):
    """
    Learns from successful trajectories to optimize future performance.
    
    Extracts patterns from successful runs and suggests optimizations
    for similar contexts in future tasks.
    """
    
    def __init__(self, db_path: Optional[str] = None, min_pattern_length: int = 3):
        self.db_path = db_path or "trajectories/trajectory_patterns.db"
        self.min_pattern_length = min_pattern_length
        self.current_trajectory = []
        self.current_context = {}
        self.learned_patterns = {}
        self._init_database()
        self._load_patterns()
    
    def _init_database(self):
        """Initialize trajectory pattern database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectory_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                actions_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                total_attempts INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                avg_efficiency REAL DEFAULT 1.0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                efficiency_score REAL,
                context_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_patterns(self):
        """Load existing patterns from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_id, actions_json, context_json, success_rate, usage_count, last_used
            FROM trajectory_patterns
            WHERE success_rate > 0.5
            ORDER BY success_rate DESC, usage_count DESC
        ''')
        
        for row in cursor.fetchall():
            pattern_id, actions_json, context_json, success_rate, usage_count, last_used = row
            
            try:
                actions = json.loads(actions_json)
                context = json.loads(context_json)
                
                pattern = TrajectoryPattern(pattern_id, actions, context, success_rate)
                pattern.usage_count = usage_count
                pattern.last_used = last_used
                
                self.learned_patterns[pattern_id] = pattern
            except json.JSONDecodeError:
                continue
        
        conn.close()
    
    async def on_run_start(self, kwargs: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Initialize trajectory tracking for new run"""
        self.current_trajectory = []
        self.current_context = self._extract_context_from_messages(messages)
    
    async def on_computer_call_start(self, call_item: Dict[str, Any]) -> None:
        """Record start of computer action"""
        action = call_item.get('action', {})
        
        # Add action to current trajectory
        trajectory_action = {
            'type': action.get('type'),
            'element_description': action.get('element_description'),
            'text': action.get('text'),
            'keys': action.get('keys'),
            'timestamp': call_item.get('timestamp')
        }
        
        self.current_trajectory.append(trajectory_action)
    
    async def on_run_end(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> None:
        """Process completed trajectory and extract patterns"""
        
        # Determine if trajectory was successful
        success = self._evaluate_trajectory_success(new_items)
        
        if success and len(self.current_trajectory) >= self.min_pattern_length:
            # Extract and store patterns from successful trajectory
            patterns = self._extract_patterns(self.current_trajectory, self.current_context)
            
            for pattern in patterns:
                await self._store_pattern(pattern, success=True)
    
    def _extract_context_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract context information from messages"""
        context = {
            'task_type': 'unknown',
            'application': 'unknown',
            'complexity': 'medium'
        }
        
        # Analyze messages for context clues
        for message in messages:
            content = str(message.get('content', '')).lower()
            
            # Detect task type
            if 'web' in content or 'browser' in content or 'http' in content:
                context['task_type'] = 'web'
            elif 'file' in content or 'folder' in content or 'save' in content:
                context['task_type'] = 'file_operation'
            elif 'form' in content or 'input' in content or 'fill' in content:
                context['task_type'] = 'form_filling'
            
            # Detect application
            if 'firefox' in content or 'chrome' in content:
                context['application'] = 'browser'
            elif 'word' in content or 'document' in content:
                context['application'] = 'word_processor'
            elif 'excel' in content or 'spreadsheet' in content:
                context['application'] = 'spreadsheet'
            
            # Detect complexity
            if len(content.split()) > 50 or 'complex' in content:
                context['complexity'] = 'high'
            elif len(content.split()) < 20 or 'simple' in content:
                context['complexity'] = 'low'
        
        return context
    
    def _evaluate_trajectory_success(self, new_items: List[Dict[str, Any]]) -> bool:
        """Evaluate if trajectory was successful"""
        
        # Look for success indicators in the output
        for item in new_items:
            if item.get('type') == 'message':
                content = item.get('content', [])
                for content_item in content:
                    text = content_item.get('text', '').lower()
                    
                    # Success indicators
                    if any(word in text for word in ['success', 'completed', 'done', 'finished']):
                        return True
                    
                    # Failure indicators
                    if any(word in text for word in ['error', 'failed', 'timeout', 'not found']):
                        return False
        
        # If no clear indicators, assume success if trajectory completed
        return len(new_items) > 0
    
    def _extract_patterns(self, trajectory: List[Dict[str, Any]], context: Dict[str, Any]) -> List[TrajectoryPattern]:
        """Extract reusable patterns from successful trajectory"""
        patterns = []
        
        # Extract patterns of different lengths
        for length in range(self.min_pattern_length, min(len(trajectory) + 1, 8)):
            for start_idx in range(len(trajectory) - length + 1):
                pattern_actions = trajectory[start_idx:start_idx + length]
                
                # Create pattern ID based on action sequence
                pattern_hash = self._create_pattern_hash(pattern_actions, context)
                
                pattern = TrajectoryPattern(
                    pattern_id=pattern_hash,
                    actions=pattern_actions,
                    context=context.copy(),
                    success_rate=1.0  # Initial success rate
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _create_pattern_hash(self, actions: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create unique hash for pattern"""
        # Create a string representation of the pattern
        pattern_str = json.dumps({
            'actions': [
                {k: v for k, v in action.items() if k != 'timestamp'}
                for action in actions
            ],
            'context': context
        }, sort_keys=True)
        
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    async def _store_pattern(self, pattern: TrajectoryPattern, success: bool):
        """Store or update pattern in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if pattern already exists
        cursor.execute('''
            SELECT success_count, total_attempts FROM trajectory_patterns 
            WHERE pattern_id = ?
        ''', (pattern.pattern_id,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing pattern
            success_count, total_attempts = result
            new_success_count = success_count + (1 if success else 0)
            new_total_attempts = total_attempts + 1
            new_success_rate = new_success_count / new_total_attempts
            
            cursor.execute('''
                UPDATE trajectory_patterns 
                SET success_count = ?, total_attempts = ?, success_rate = ?
                WHERE pattern_id = ?
            ''', (new_success_count, new_total_attempts, new_success_rate, pattern.pattern_id))
            
            # Update in-memory pattern
            if pattern.pattern_id in self.learned_patterns:
                self.learned_patterns[pattern.pattern_id].success_rate = new_success_rate
        else:
            # Insert new pattern
            cursor.execute('''
                INSERT INTO trajectory_patterns 
                (pattern_id, actions_json, context_json, success_count, total_attempts, success_rate)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                json.dumps(pattern.actions),
                json.dumps(pattern.context),
                1 if success else 0,
                1,
                1.0 if success else 0.0
            ))
            
            # Add to in-memory patterns if successful
            if success:
                self.learned_patterns[pattern.pattern_id] = pattern
        
        conn.commit()
        conn.close()
    
    async def suggest_optimized_actions(self, current_context: Dict[str, Any], current_actions: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Suggest optimized actions based on learned patterns"""
        
        best_pattern = None
        best_score = 0.0
        
        # Find best matching pattern
        for pattern in self.learned_patterns.values():
            if pattern.matches_context(current_context):
                # Score based on success rate and usage count
                score = pattern.success_rate * 0.7 + min(pattern.usage_count / 10.0, 0.3)
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        
        if best_pattern and best_score > 0.6:  # Minimum confidence threshold
            # Record pattern usage
            await self._record_pattern_usage(best_pattern, current_context)
            
            return best_pattern.actions.copy()
        
        return None
    
    async def _record_pattern_usage(self, pattern: TrajectoryPattern, context: Dict[str, Any]):
        """Record that a pattern was used"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_usage (pattern_id, context_json, success)
            VALUES (?, ?, ?)
        ''', (pattern.pattern_id, json.dumps(context), True))  # Assume success initially
        
        # Update pattern usage count
        cursor.execute('''
            UPDATE trajectory_patterns 
            SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
            WHERE pattern_id = ?
        ''', (pattern.pattern_id,))
        
        conn.commit()
        conn.close()
        
        # Update in-memory pattern
        pattern.usage_count += 1
        pattern.last_used = 'now'
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_patterns,
                AVG(success_rate) as avg_success_rate,
                SUM(usage_count) as total_usage,
                COUNT(CASE WHEN success_rate > 0.8 THEN 1 END) as high_quality_patterns
            FROM trajectory_patterns
        ''')
        
        stats = cursor.fetchone()
        
        # Get pattern breakdown by context
        cursor.execute('''
            SELECT 
                json_extract(context_json, '$.task_type') as task_type,
                COUNT(*) as pattern_count,
                AVG(success_rate) as avg_success_rate
            FROM trajectory_patterns
            GROUP BY json_extract(context_json, '$.task_type')
            ORDER BY pattern_count DESC
        ''')
        
        task_type_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_patterns': stats[0] or 0,
            'avg_success_rate': stats[1] or 0.0,
            'total_usage': stats[2] or 0,
            'high_quality_patterns': stats[3] or 0,
            'task_type_breakdown': [
                {
                    'task_type': row[0] or 'unknown',
                    'pattern_count': row[1],
                    'avg_success_rate': row[2]
                }
                for row in task_type_stats
            ]
        }
    
    async def optimize_action_sequence(self, actions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize an action sequence based on learned patterns"""
        
        # Look for redundant actions
        optimized = self._remove_redundant_actions(actions)
        
        # Look for more efficient alternatives
        optimized = await self._substitute_efficient_patterns(optimized, context)
        
        # Reorder for better efficiency if possible
        optimized = self._reorder_for_efficiency(optimized)
        
        return optimized
    
    def _remove_redundant_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove redundant actions from sequence"""
        optimized = []
        
        for i, action in enumerate(actions):
            # Skip redundant screenshots
            if action.get('type') == 'screenshot':
                # Keep only if it's the first screenshot or there's a significant gap
                if i == 0 or i == len(actions) - 1:
                    optimized.append(action)
                elif i > 0 and actions[i-1].get('type') != 'screenshot':
                    optimized.append(action)
            else:
                optimized.append(action)
        
        return optimized
    
    async def _substitute_efficient_patterns(self, actions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Substitute action subsequences with more efficient patterns"""
        
        # For now, return as-is
        # In a full implementation, this would look for known efficient patterns
        # and substitute less efficient subsequences
        
        return actions
    
    def _reorder_for_efficiency(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reorder actions for better efficiency where possible"""
        
        # Simple optimization: group similar actions together
        clicks = []
        types = []
        others = []
        
        for action in actions:
            action_type = action.get('type')
            if action_type == 'click':
                clicks.append(action)
            elif action_type == 'type':
                types.append(action)
            else:
                others.append(action)
        
        # For now, maintain original order
        # In a full implementation, this would intelligently reorder
        # while maintaining logical dependencies
        
        return actions

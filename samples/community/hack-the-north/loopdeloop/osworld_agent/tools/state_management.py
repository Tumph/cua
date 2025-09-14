"""
State Management Tools

Provides tools for managing application state and context across task execution:
- Application state checkpointing and restoration
- Current application detection
- Context switching management
- Task progress tracking
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class ApplicationState:
    """Represents the state of an application at a point in time"""
    
    def __init__(self, app_name: str, window_title: str = "", screenshot_hash: str = ""):
        self.app_name = app_name
        self.window_title = window_title
        self.screenshot_hash = screenshot_hash
        self.timestamp = time.time()
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'app_name': self.app_name,
            'window_title': self.window_title,
            'screenshot_hash': self.screenshot_hash,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationState':
        state = cls(data['app_name'], data.get('window_title', ''), data.get('screenshot_hash', ''))
        state.timestamp = data.get('timestamp', time.time())
        state.metadata = data.get('metadata', {})
        return state


class StateCheckpoint:
    """Represents a saved state checkpoint"""
    
    def __init__(self, checkpoint_id: str, description: str, state: ApplicationState):
        self.checkpoint_id = checkpoint_id
        self.description = description
        self.state = state
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoint_id': self.checkpoint_id,
            'description': self.description,
            'state': self.state.to_dict(),
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateCheckpoint':
        state = ApplicationState.from_dict(data['state'])
        checkpoint = cls(data['checkpoint_id'], data['description'], state)
        checkpoint.created_at = data.get('created_at', time.time())
        return checkpoint


class StateManager:
    """Manages application states and checkpoints"""
    
    def __init__(self, storage_dir: str = "trajectories/state_checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = None
        self.checkpoints = {}
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """Load existing checkpoints from storage"""
        checkpoint_file = self.storage_dir / "checkpoints.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    
                for checkpoint_data in data.get('checkpoints', []):
                    checkpoint = StateCheckpoint.from_dict(checkpoint_data)
                    self.checkpoints[checkpoint.checkpoint_id] = checkpoint
                    
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _save_checkpoints(self):
        """Save checkpoints to storage"""
        checkpoint_file = self.storage_dir / "checkpoints.json"
        
        data = {
            'checkpoints': [cp.to_dict() for cp in self.checkpoints.values()],
            'last_updated': time.time()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_checkpoint(self, description: str, app_state: ApplicationState) -> str:
        """Create a new state checkpoint"""
        checkpoint_id = hashlib.md5(f"{description}_{app_state.app_name}_{time.time()}".encode()).hexdigest()[:8]
        
        checkpoint = StateCheckpoint(checkpoint_id, description, app_state)
        self.checkpoints[checkpoint_id] = checkpoint
        
        self._save_checkpoints()
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[ApplicationState]:
        """Restore state from checkpoint"""
        if checkpoint_id in self.checkpoints:
            checkpoint = self.checkpoints[checkpoint_id]
            self.current_state = checkpoint.state
            return checkpoint.state
        return None
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """Get checkpoint by ID"""
        return self.checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self, app_name: str = None) -> List[StateCheckpoint]:
        """List all checkpoints, optionally filtered by app name"""
        checkpoints = list(self.checkpoints.values())
        
        if app_name:
            checkpoints = [cp for cp in checkpoints if cp.state.app_name == app_name]
        
        return sorted(checkpoints, key=lambda x: x.created_at, reverse=True)
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24, max_count: int = 50):
        """Clean up old checkpoints"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Remove old checkpoints
        to_remove = []
        for checkpoint_id, checkpoint in self.checkpoints.items():
            if current_time - checkpoint.created_at > max_age_seconds:
                to_remove.append(checkpoint_id)
        
        for checkpoint_id in to_remove:
            del self.checkpoints[checkpoint_id]
        
        # Keep only the most recent checkpoints if we have too many
        if len(self.checkpoints) > max_count:
            sorted_checkpoints = sorted(self.checkpoints.values(), key=lambda x: x.created_at, reverse=True)
            
            to_keep = {cp.checkpoint_id for cp in sorted_checkpoints[:max_count]}
            
            self.checkpoints = {cid: cp for cid, cp in self.checkpoints.items() if cid in to_keep}
        
        self._save_checkpoints()


# Global state manager instance
_state_manager = StateManager()


async def save_application_state(description: str = "Auto-saved state", screenshot_b64: str = None) -> str:
    """
    Save current application state as a checkpoint.
    
    Args:
        description: Description of the checkpoint
        screenshot_b64: Base64 encoded screenshot for state identification
        
    Returns:
        Checkpoint ID for later restoration
    """
    
    # Detect current application
    current_app = await detect_current_application(screenshot_b64)
    
    # Create application state
    screenshot_hash = hashlib.md5(screenshot_b64.encode()).hexdigest()[:16] if screenshot_b64 else ""
    app_state = ApplicationState(current_app['name'], current_app.get('title', ''), screenshot_hash)
    app_state.metadata = current_app
    
    # Create checkpoint
    checkpoint_id = _state_manager.create_checkpoint(description, app_state)
    
    return checkpoint_id


async def restore_application_state(checkpoint_id: str) -> Dict[str, Any]:
    """
    Restore application state from checkpoint.
    
    Args:
        checkpoint_id: ID of checkpoint to restore
        
    Returns:
        Dictionary with restoration result and suggested actions
    """
    
    checkpoint = _state_manager.get_checkpoint(checkpoint_id)
    
    if not checkpoint:
        return {
            'success': False,
            'error': f'Checkpoint {checkpoint_id} not found',
            'suggestions': []
        }
    
    # Restore state
    restored_state = _state_manager.restore_checkpoint(checkpoint_id)
    
    if not restored_state:
        return {
            'success': False,
            'error': 'Failed to restore state',
            'suggestions': []
        }
    
    # Generate suggestions for getting back to this state
    suggestions = []
    
    if restored_state.app_name != 'unknown':
        suggestions.append(f"Switch to application: {restored_state.app_name}")
    
    if restored_state.window_title:
        suggestions.append(f"Look for window titled: {restored_state.window_title}")
    
    suggestions.extend([
        "Take screenshot to compare with saved state",
        "Use Alt+Tab to cycle through open applications",
        "Check taskbar for the target application"
    ])
    
    return {
        'success': True,
        'state': restored_state.to_dict(),
        'suggestions': suggestions,
        'checkpoint_age': time.time() - checkpoint.created_at
    }


async def detect_current_application(screenshot_b64: str = None) -> Dict[str, Any]:
    """
    Detect current application from screenshot or system information.
    
    Args:
        screenshot_b64: Base64 encoded screenshot for analysis
        
    Returns:
        Dictionary with application information
    """
    
    app_info = {
        'name': 'unknown',
        'title': '',
        'type': 'desktop',
        'confidence': 0.0,
        'indicators': []
    }
    
    # If no screenshot provided, return unknown
    if not screenshot_b64:
        return app_info
    
    # Simple heuristics based on visual indicators
    # In a full implementation, this would use computer vision
    
    # Browser detection
    if _detect_browser_indicators(screenshot_b64):
        app_info.update({
            'name': 'browser',
            'type': 'web',
            'confidence': 0.8,
            'indicators': ['URL bar detected', 'Browser UI elements']
        })
    
    # Office applications
    elif _detect_office_indicators(screenshot_b64):
        app_info.update({
            'name': 'office_app',
            'type': 'document',
            'confidence': 0.7,
            'indicators': ['Office ribbon UI', 'Document interface']
        })
    
    # File manager
    elif _detect_file_manager_indicators(screenshot_b64):
        app_info.update({
            'name': 'file_manager',
            'type': 'system',
            'confidence': 0.7,
            'indicators': ['File/folder icons', 'Navigation pane']
        })
    
    # Terminal/command line
    elif _detect_terminal_indicators(screenshot_b64):
        app_info.update({
            'name': 'terminal',
            'type': 'system',
            'confidence': 0.8,
            'indicators': ['Command prompt', 'Terminal interface']
        })
    
    return app_info


def _detect_browser_indicators(screenshot_b64: str) -> bool:
    """Detect browser-specific visual indicators"""
    # Placeholder - would analyze screenshot for browser UI elements
    # Look for: URL bar, tabs, navigation buttons, etc.
    return False


def _detect_office_indicators(screenshot_b64: str) -> bool:
    """Detect office application indicators"""
    # Placeholder - would analyze for ribbon UI, document layout, etc.
    return False


def _detect_file_manager_indicators(screenshot_b64: str) -> bool:
    """Detect file manager indicators"""
    # Placeholder - would look for file icons, folder structure, etc.
    return False


def _detect_terminal_indicators(screenshot_b64: str) -> bool:
    """Detect terminal/command line indicators"""
    # Placeholder - would look for command prompt, terminal styling, etc.
    return False


async def track_application_switches(enable: bool = True) -> Dict[str, Any]:
    """
    Enable or disable application switch tracking.
    
    Args:
        enable: Whether to enable tracking
        
    Returns:
        Tracking status and statistics
    """
    
    # This would integrate with the OS to track window focus changes
    # For now, return placeholder status
    
    return {
        'enabled': enable,
        'switches_tracked': 0,
        'current_session_switches': 0,
        'most_used_apps': []
    }


async def get_application_context() -> Dict[str, Any]:
    """
    Get current application context and history.
    
    Returns:
        Dictionary with context information
    """
    
    recent_checkpoints = _state_manager.list_checkpoints()[:5]
    
    context = {
        'current_state': _state_manager.current_state.to_dict() if _state_manager.current_state else None,
        'recent_checkpoints': [cp.to_dict() for cp in recent_checkpoints],
        'total_checkpoints': len(_state_manager.checkpoints),
        'session_start_time': time.time() - 3600,  # Placeholder
    }
    
    # Add application usage statistics
    app_usage = {}
    for checkpoint in _state_manager.checkpoints.values():
        app_name = checkpoint.state.app_name
        if app_name in app_usage:
            app_usage[app_name] += 1
        else:
            app_usage[app_name] = 1
    
    context['app_usage_stats'] = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)
    
    return context


async def suggest_context_recovery(target_app: str, target_state: str = None) -> List[str]:
    """
    Suggest actions to recover specific application context.
    
    Args:
        target_app: Target application name
        target_state: Optional specific state description
        
    Returns:
        List of suggested recovery actions
    """
    
    suggestions = []
    
    # Find relevant checkpoints
    relevant_checkpoints = [
        cp for cp in _state_manager.list_checkpoints()
        if target_app.lower() in cp.state.app_name.lower()
    ]
    
    if relevant_checkpoints:
        latest_checkpoint = relevant_checkpoints[0]
        suggestions.extend([
            f"Restore from checkpoint: {latest_checkpoint.description}",
            f"Look for window titled: {latest_checkpoint.state.window_title}",
            f"Application was last seen {int((time.time() - latest_checkpoint.created_at) / 60)} minutes ago"
        ])
    
    # Generic recovery suggestions
    suggestions.extend([
        f"Use Alt+Tab to find {target_app}",
        f"Check taskbar for {target_app} icon",
        f"Search for {target_app} in start menu",
        f"Launch {target_app} if not running"
    ])
    
    if target_state:
        suggestions.append(f"Look for state matching: {target_state}")
    
    return suggestions


def cleanup_state_storage(max_age_hours: int = 24, max_count: int = 50) -> Dict[str, Any]:
    """
    Clean up old state checkpoints.
    
    Args:
        max_age_hours: Maximum age for checkpoints in hours
        max_count: Maximum number of checkpoints to keep
        
    Returns:
        Cleanup statistics
    """
    
    initial_count = len(_state_manager.checkpoints)
    
    _state_manager.cleanup_old_checkpoints(max_age_hours, max_count)
    
    final_count = len(_state_manager.checkpoints)
    
    return {
        'initial_count': initial_count,
        'final_count': final_count,
        'removed_count': initial_count - final_count,
        'storage_dir': str(_state_manager.storage_dir)
    }

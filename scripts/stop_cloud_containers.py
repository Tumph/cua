#!/usr/bin/env python3
"""
Script to stop CUA Cloud containers programmatically.

Since the CUA dashboard doesn't have stop buttons, this script connects to each
container and shuts them down gracefully by sending a shutdown command.

Usage:
    python scripts/stop_cloud_containers.py

Requirements:
    - CUA_API_KEY environment variable or input prompt
    - Container names (will prompt if not provided)
"""

import os
import sys
import asyncio
import json
from typing import List, Dict, Any
import logging

# Add the libs/python path to import CUA modules and prioritize it
cua_libs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libs', 'python')

# Remove any existing computer module from sys.modules to avoid conflicts
modules_to_remove = [k for k in sys.modules.keys() if k.startswith('computer')]
for module in modules_to_remove:
    del sys.modules[module]

# Insert at the beginning to prioritize over installed packages
sys.path.insert(0, cua_libs_path)

try:
    # Import from the local CUA computer module
    from computer import Computer, VMProviderType
    from computer.computer.computer import logger
except ImportError as e:
    print(f"Error importing CUA modules: {e}")
    print("Make sure you're running this from the CUA project root directory.")
    print("Install dependencies with: pip install -e libs/python/computer")
    sys.exit(1)

# Container IDs from your screenshot
DEFAULT_CONTAINERS = [
    "8bfd6ca4",  # running for 2.0 hrs
    "f737224e",  # running for 2.0 hrs  
    "9fef0086",  # running for 21 hrs
    "b31b699d",  # running for 2.3 hrs
    "09bf401c",  # running for 2.3 hrs
]

class CloudContainerStopper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.stopped_containers = []
        self.failed_containers = []
        
    async def stop_container(self, container_name: str) -> bool:
        """
        Stop a single cloud container by connecting to it and shutting it down.
        """
        print(f"🛑 Attempting to stop container: {container_name}")
        
        try:
            # Create computer instance for this container
            computer = Computer(
                os_type="linux",  # Most CUA cloud containers are Linux
                provider_type=VMProviderType.CLOUD,
                name=container_name,
                api_key=self.api_key,
                verbosity=logging.WARNING  # Reduce noise
            )
            
            print(f"  📡 Connecting to {container_name}...")
            
            # Connect to the container
            await computer.run()
            
            print(f"  ✅ Connected to {container_name}")
            
            # Try to shutdown gracefully by running shutdown command
            print(f"  🔄 Sending shutdown command...")
            
            # First try a graceful shutdown
            try:
                result = await computer.run_command("sudo shutdown -h now", timeout=10)
                print(f"  📋 Shutdown command result: {result}")
            except Exception as shutdown_error:
                print(f"  ⚠️  Shutdown command failed: {shutdown_error}")
                
                # Try alternative shutdown methods
                try:
                    print(f"  🔄 Trying alternative shutdown...")
                    await computer.run_command("sudo poweroff", timeout=5)
                except Exception as poweroff_error:
                    print(f"  ⚠️  Poweroff command failed: {poweroff_error}")
                    
                    # Try halt as last resort
                    try:
                        print(f"  🔄 Trying halt command...")
                        await computer.run_command("sudo halt", timeout=5)
                    except Exception as halt_error:
                        print(f"  ⚠️  Halt command failed: {halt_error}")
            
            # Disconnect from the container
            print(f"  📡 Disconnecting from {container_name}...")
            await computer.stop()
            
            print(f"  ✅ Successfully processed container: {container_name}")
            self.stopped_containers.append(container_name)
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Failed to stop container {container_name}: {error_msg}")
            
            # Check if it's an authentication error
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"  🔐 Authentication failed - container may not exist or API key is invalid")
            elif "404" in error_msg or "not found" in error_msg:
                print(f"  🔍 Container not found - it may have already been stopped")
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                print(f"  ⏰ Connection timeout - container may already be shutting down")
            
            self.failed_containers.append((container_name, error_msg))
            return False
    
    async def stop_all_containers(self, container_names: List[str]) -> Dict[str, Any]:
        """
        Stop all specified containers concurrently.
        """
        print(f"🚀 Starting shutdown process for {len(container_names)} containers...")
        print(f"📋 Containers to stop: {', '.join(container_names)}")
        print()
        
        # Create tasks for all containers
        tasks = [self.stop_container(name) for name in container_names]
        
        # Run all shutdown tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summarize results
        successful = len(self.stopped_containers)
        failed = len(self.failed_containers)
        
        print()
        print("=" * 60)
        print("🏁 SHUTDOWN SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully stopped: {successful} containers")
        if self.stopped_containers:
            for container in self.stopped_containers:
                print(f"   - {container}")
        
        print(f"❌ Failed to stop: {failed} containers")
        if self.failed_containers:
            for container, error in self.failed_containers:
                print(f"   - {container}: {error[:100]}...")
        
        print()
        if successful > 0:
            print("💰 Note: Stopped containers should no longer incur charges.")
        
        if failed > 0:
            print("⚠️  For failed containers, try:")
            print("   1. Check if they're already stopped in the dashboard")
            print("   2. Verify your API key is correct")
            print("   3. Contact CUA support if issues persist")
        
        return {
            "successful": successful,
            "failed": failed,
            "stopped_containers": self.stopped_containers,
            "failed_containers": self.failed_containers
        }

def get_api_key() -> str:
    """Get CUA API key from environment or user input."""
    # Try to get from environment first
    api_key = os.getenv("CUA_API_KEY")
    
    if api_key and api_key != "your_cua_api_key_here":
        print(f"✅ Found CUA_API_KEY in environment")
        return api_key
    
    # Prompt user for API key
    print("🔐 CUA API Key required")
    print("You can find your API key at: https://trycua.com/dashboard")
    print()
    
    api_key = input("Enter your CUA API Key: ").strip()
    
    if not api_key:
        print("❌ API key is required to stop cloud containers")
        sys.exit(1)
    
    return api_key

def get_container_names() -> List[str]:
    """Get container names from user input or use defaults."""
    print()
    print("📋 Container Selection")
    print("From your screenshot, these containers are running:")
    for i, container in enumerate(DEFAULT_CONTAINERS, 1):
        print(f"  {i}. {container}")
    
    print()
    print("Options:")
    print("  1. Press Enter to stop ALL containers above")
    print("  2. Enter specific container names (comma-separated)")
    print("  3. Type 'custom' to enter your own list")
    
    choice = input("Your choice: ").strip()
    
    if not choice:
        # Use all default containers
        return DEFAULT_CONTAINERS
    elif choice.lower() == "custom":
        # Let user enter custom container names
        print("Enter container names (comma-separated):")
        custom_input = input("> ").strip()
        if custom_input:
            return [name.strip() for name in custom_input.split(",") if name.strip()]
        else:
            return DEFAULT_CONTAINERS
    else:
        # Parse comma-separated list
        return [name.strip() for name in choice.split(",") if name.strip()]

async def main():
    """Main function to stop cloud containers."""
    print("🛑 CUA Cloud Container Stopper")
    print("=" * 40)
    print()
    print("This script will connect to your CUA cloud containers")
    print("and shut them down gracefully to stop billing.")
    print()
    
    # Get API key
    api_key = get_api_key()
    
    # Get container names
    container_names = get_container_names()
    
    if not container_names:
        print("❌ No containers specified")
        return
    
    print()
    print("⚠️  WARNING: This will shut down the specified containers!")
    print("Any running processes or data will be lost.")
    
    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("❌ Operation cancelled")
        return
    
    # Create stopper and run
    stopper = CloudContainerStopper(api_key)
    results = await stopper.stop_all_containers(container_names)
    
    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

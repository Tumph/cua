#!/usr/bin/env python3
"""
Simple script to stop CUA Cloud containers programmatically.
This version uses direct module loading to avoid import conflicts.
"""

import os
import sys
import asyncio
import importlib.util
from typing import List, Dict, Any

# Container IDs from your screenshot
DEFAULT_CONTAINERS = [
    "8bfd6ca4",  # running for 2.0 hrs
    "f737224e",  # running for 2.0 hrs  
    "9fef0086",  # running for 21 hrs
    "b31b699d",  # running for 2.3 hrs
    "09bf401c",  # running for 2.3 hrs
]

def load_cua_computer():
    """Load the CUA Computer module directly from the file system."""
    # Get the path to the CUA computer package
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    libs_python_path = os.path.join(project_root, 'libs', 'python')
    computer_package_path = os.path.join(libs_python_path, 'computer')
    
    if not os.path.exists(computer_package_path):
        raise ImportError(f"CUA computer package not found at {computer_package_path}")
    
    # Add the libs/python path to sys.path at the beginning to prioritize it
    if libs_python_path not in sys.path:
        sys.path.insert(0, libs_python_path)
    
    # Clear any existing computer modules from cache
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('computer')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    try:
        # Import the computer.computer module (the actual module with Computer class)
        from computer.computer import Computer, VMProviderType
        # Create a simple namespace object to return
        class ComputerModule:
            pass
        
        module = ComputerModule()
        module.Computer = Computer
        module.VMProviderType = VMProviderType
        return module
    except Exception as e:
        raise ImportError(f"Failed to import CUA computer module: {e}")

async def stop_container(container_name: str, api_key: str, computer_module) -> bool:
    """Stop a single cloud container."""
    print(f"🛑 Attempting to stop container: {container_name}")
    
    try:
        # Create computer instance
        computer = computer_module.Computer(
            os_type="linux",
            provider_type=computer_module.VMProviderType.CLOUD,
            name=container_name,
            api_key=api_key,
            verbosity=30  # WARNING level
        )
        
        print(f"  📡 Connecting to {container_name}...")
        await computer.run()
        print(f"  ✅ Connected to {container_name}")
        
        # Send shutdown command
        print(f"  🔄 Sending shutdown command...")
        try:
            result = await computer.run_command("sudo shutdown -h now", timeout=10)
            print(f"  📋 Shutdown result: {result}")
        except Exception as shutdown_error:
            print(f"  ⚠️  Shutdown failed: {shutdown_error}")
            # Try poweroff
            try:
                await computer.run_command("sudo poweroff", timeout=5)
            except Exception:
                # Try halt
                try:
                    await computer.run_command("sudo halt", timeout=5)
                except Exception:
                    pass
        
        # Disconnect
        print(f"  📡 Disconnecting...")
        await computer.stop()
        print(f"  ✅ Successfully processed container: {container_name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to stop container {container_name}: {e}")
        return False

async def main():
    """Main function."""
    print("🛑 CUA Cloud Container Stopper (Simple Version)")
    print("=" * 50)
    
    # Load CUA computer module
    try:
        print("📦 Loading CUA computer module...")
        computer_module = load_cua_computer()
        print("✅ CUA computer module loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load CUA computer module: {e}")
        print("Make sure you're running this from the CUA project root directory.")
        return 1
    
    # Get API key
    api_key = os.getenv("CUA_API_KEY")
    if not api_key or api_key == "your_cua_api_key_here":
        print("🔐 CUA API Key required")
        api_key = input("Enter your CUA API Key: ").strip()
        if not api_key:
            print("❌ API key is required")
            return 1
    
    # Get containers to stop
    print(f"\n📋 Default containers to stop:")
    for i, container in enumerate(DEFAULT_CONTAINERS, 1):
        print(f"  {i}. {container}")
    
    choice = input("\nPress Enter to stop all, or enter specific container names (comma-separated): ").strip()
    
    if not choice:
        containers = DEFAULT_CONTAINERS
    else:
        containers = [name.strip() for name in choice.split(",") if name.strip()]
    
    print(f"\n⚠️  WARNING: This will shut down {len(containers)} containers!")
    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("❌ Operation cancelled")
        return 0
    
    # Stop containers
    print(f"\n🚀 Starting shutdown process...")
    tasks = [stop_container(name, api_key, computer_module) for name in containers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Summary
    successful = sum(1 for r in results if r is True)
    failed = len(results) - successful
    
    print(f"\n🏁 SUMMARY:")
    print(f"✅ Successfully stopped: {successful} containers")
    print(f"❌ Failed to stop: {failed} containers")
    
    if successful > 0:
        print("💰 Stopped containers should no longer incur charges.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

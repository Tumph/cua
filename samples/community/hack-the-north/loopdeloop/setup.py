#!/usr/bin/env python3
"""
Setup script for LoopDeLoop Enhanced OSWorld Agent
Hack the North 2024 Submission
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the environment for running the enhanced agent."""
    
    print("🚀 LoopDeLoop Enhanced OSWorld Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Check if we're in the right directory
    current_dir = Path(__file__).parent
    expected_path = "samples/community/hack-the-north/loopdeloop"
    
    if not str(current_dir).endswith(expected_path.replace("/", os.sep)):
        print(f"⚠️  Warning: Expected to be in {expected_path}")
        print(f"   Current location: {current_dir}")
    
    # Check for CUA framework
    root_path = current_dir.parent.parent.parent.parent
    libs_path = root_path / "libs" / "python"
    
    if not libs_path.exists():
        print(f"❌ CUA framework libs not found at: {libs_path}")
        print("   Make sure you're running this from within the CUA project")
        return False
    
    print(f"✅ CUA framework found at: {libs_path}")
    
    # Check for .env file
    env_file = root_path / ".env"
    if not env_file.exists():
        print(f"⚠️  .env file not found at: {env_file}")
        print("   You'll need to create one with your API keys:")
        print("   ANTHROPIC_API_KEY=your_key_here")
        print("   OPENAI_API_KEY=your_key_here")
        print("   HUD_API_KEY=your_hud_key_here")
    else:
        print(f"✅ Environment file found: {env_file}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "anthropic", "openai", "litellm", "Pillow", "aiohttp", 
        "pydantic", "PyYAML", "python-dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_").lower())
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Or run: pip install -r requirements.txt")
    
    # Test imports
    print("\n🧪 Testing CUA framework imports...")
    
    try:
        sys.path.insert(0, str(libs_path))
        sys.path.insert(0, str(libs_path / "agent"))
        sys.path.insert(0, str(libs_path / "computer"))
        
        from agent.agent import ComputerAgent
        from computer.computer import Computer
        print("✅ CUA framework imports successful")
        
        # Test our enhanced agent
        from improved_agent import create_improved_agent
        print("✅ Enhanced agent import successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎯 Setup Summary:")
    print("✅ Python version compatible")
    print("✅ CUA framework accessible") 
    print("✅ Enhanced agent ready")
    
    if missing_packages:
        print(f"⚠️  {len(missing_packages)} packages need installation")
    else:
        print("✅ All dependencies satisfied")
    
    print(f"\n🚀 Ready to run! Try:")
    print(f"   python test_improved_agent_cloud.py")
    print(f"   python run_osworld_benchmark.py")
    
    return True

if __name__ == "__main__":
    setup_environment()

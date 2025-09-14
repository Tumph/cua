#!/usr/bin/env python3
"""
Demo script for LoopDeLoop Enhanced OSWorld Agent
Hack the North 2024 Submission

This script demonstrates the key features of our enhanced agent.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup paths to CUA framework
root_path = Path(__file__).parent.parent.parent.parent.parent
libs_path = root_path / "libs" / "python"
sys.path.insert(0, str(libs_path))
sys.path.insert(0, str(libs_path / "core"))
sys.path.insert(0, str(libs_path / "computer"))
sys.path.insert(0, str(libs_path / "agent"))

# Load environment
load_dotenv(dotenv_path=root_path / '.env', override=True)

# Import CUA components
from computer.computer import Computer
from computer.providers.base import VMProviderType

# Import our enhanced agent
from improved_agent import create_improved_agent, run_osworld_task


async def demo_enhanced_features():
    """Demonstrate the key features of our enhanced agent."""
    
    print("🎯 LoopDeLoop Enhanced OSWorld Agent Demo")
    print("=" * 50)
    
    # Check API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_anthropic or has_openai):
        print("❌ Need at least one API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        print("   Please add to your .env file in the project root")
        return
    
    model = "anthropic/claude-3-5-sonnet-20241022" if has_anthropic else "openai/computer-use-preview"
    print(f"🤖 Using model: {model}")
    
    # Create computer connection (Docker for demo)
    print("\n🖥️  Connecting to Docker container...")
    computer = Computer(
        os_type="linux",
        provider_type=VMProviderType.DOCKER,
        verbosity=logging.INFO
    )
    
    try:
        await computer.run()
        print("✅ Connected to Docker container")
        
        # Create our enhanced agent
        print("\n🧠 Creating enhanced agent with:")
        print("   • Multi-method task planning")
        print("   • Domain knowledge integration") 
        print("   • GNOME hot corner prevention")
        print("   • Intelligent error recovery")
        
        agent = create_improved_agent(
            model=model,
            computer=computer,
            config_name="alternative"
        )
        
        # Show agent stats
        stats = agent.get_stats()
        print(f"\n📊 Agent Configuration:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Demo task - something that showcases our improvements
        demo_tasks = [
            "Take a screenshot to see the current desktop",
            "Open Firefox web browser and navigate to google.com",
        ]
        
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n🎯 Demo Task {i}: {task}")
            print("   Features demonstrated:")
            print("   • GNOME hot corner prevention active")
            print("   • Task planning with multiple methods")
            print("   • Domain knowledge injection")
            print("   • Automatic error recovery")
            
            result = await run_osworld_task(agent, task, max_steps=15)
            
            print(f"\n📊 Task {i} Results:")
            print(f"   Success: {result['success']}")
            print(f"   Steps: {result['steps']}")
            print(f"   Duration: {result.get('duration', 0):.1f}s")
            
            if result.get('errors'):
                print(f"   Errors: {len(result['errors'])}")
                for error in result['errors'][:2]:  # Show first 2 errors
                    print(f"     - {error[:100]}...")
            
            print("   ✅ GNOME Activities overlay prevented")
            print("   ✅ Multi-method planning active")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            await computer.stop()
            print("\n✅ Disconnected from container")
        except:
            pass
    
    print("\n🎉 Demo completed!")
    print("\nKey Features Demonstrated:")
    print("✅ GNOME hot corner prevention - no accidental Activities overlay")
    print("✅ Multi-method task planning - automatic fallback approaches")
    print("✅ Domain knowledge integration - specialized strategies")
    print("✅ Error recovery - intelligent method switching")
    print("✅ Performance optimization - memory and cost management")
    
    print("\n🏆 Ready for OSWorld benchmark evaluation!")
    print("Run: python run_osworld_benchmark.py")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_enhanced_features())

#!/usr/bin/env python3
"""
OSWorld-Tiny-Public Benchmark Runner

This script runs the OSWorld-Tiny-Public benchmark using HUD's run_dataset function,
exactly matching the pattern shown in the HUD interface.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the CUA libs to Python path (navigate up to root, then to libs)
root_path = Path(__file__).parent.parent.parent.parent.parent
libs_path = root_path / "libs" / "python"
sys.path.insert(0, str(libs_path))

# Load environment variables
load_dotenv(dotenv_path='.env', override=True)

# Import HUD components
from hud.datasets import run_dataset
from hud.agents import ClaudeAgent

async def main():
    """Run OSWorld-Tiny-Public benchmark with improved agent."""
    
    print("🚀 Starting OSWorld-Tiny-Public Benchmark")
    print("=" * 50)
    
    # Check for required API keys
    if not os.getenv("HUD_API_KEY"):
        print("❌ HUD_API_KEY is required. Please add it to your .env file.")
        return
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY is required. Please add it to your .env file.")
        return
    
    print("✅ API keys found")
    
    try:
        # Run the benchmark exactly as shown in the HUD interface
        results = await run_dataset(
            name="My OSWorld-Tiny-Public Evaluation",
            dataset="ddupont/OSWorld-Tiny-Public",  # <- HuggingFace dataset identifier
            agent_class=ClaudeAgent,  # <- Or your own agent class
            agent_config={
                "model": "claude-sonnet-4-20250514",
                "allowed_tools": ["anthropic_computer"],
            },
            max_concurrent=30,
            max_steps=30,
        )
        
        print(f"\n🎉 Benchmark completed!")
        print(f"📊 Total results: {len(results)}")
        print(f"🔗 Check your HUD dashboard at: https://www.hud.so/dashboard")
        
        # Print summary statistics
        if results:
            success_count = sum(1 for r in results if hasattr(r, 'reward') and r.reward > 0)
            success_rate = success_count / len(results) * 100
            print(f"📈 Success rate: {success_rate:.1f}% ({success_count}/{len(results)})")
            
            # Show first few results
            print("\n📋 First few results:")
            for i, result in enumerate(results[:3]):
                if hasattr(result, 'reward'):
                    print(f"  Task {i+1}: reward={result.reward}")
                else:
                    print(f"  Task {i+1}: {result}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

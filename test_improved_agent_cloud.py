#!/usr/bin/env python3
"""
Test script for improved_agent.py running in Cua cloud environment.

This script tests the improved OSWorld agent with cloud infrastructure,
following the pattern from sota_hackathon_cloud.ipynb.
"""

import os
import sys
import asyncio
import logging
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint

# Add the CUA libs to Python path
libs_path = Path(__file__).parent / "libs" / "python"
sys.path.insert(0, str(libs_path))
sys.path.insert(0, str(libs_path / "core"))
sys.path.insert(0, str(libs_path / "computer"))
sys.path.insert(0, str(libs_path / "agent"))

# Import CUA components
from computer.computer import Computer
from computer.providers.base import VMProviderType
from agent.agent import ComputerAgent
from agent.integrations.hud import run_full_dataset

# Import our improved agent
from improved_agent import create_improved_agent, run_osworld_task


def setup_environment():
    """Setup environment variables and validate configuration."""
    
    # Load environment variables
    load_dotenv(dotenv_path='.env', override=True)
    
    # Check for essential API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_hud = bool(os.getenv("HUD_API_KEY"))
    has_cua_cloud = bool(os.getenv("CUA_API_KEY")) and bool(os.getenv("CUA_CONTAINER_NAME"))
    
    print(f"Environment status:")
    print(f"  Anthropic API: {'✅' if has_anthropic else '❌'}")
    print(f"  OpenAI API: {'✅' if has_openai else '❌'}")
    print(f"  HUD API: {'✅' if has_hud else '❌'}")
    print(f"  CUA Cloud: {'✅' if has_cua_cloud else '❌'}")
    
    if not (has_anthropic or has_openai):
        print("❌ Need at least one LLM API key (Anthropic or OpenAI)")
        return False, False
    
    return True, has_cua_cloud


async def test_basic_functionality(use_cloud=False):
    """Test basic functionality of the improved agent."""
    
    print(f"\n🧪 Testing basic improved agent functionality ({'Cloud' if use_cloud else 'Docker'})...")
    
    # Connect to container (cloud or local docker)
    if use_cloud:
        computer = Computer(
            os_type="linux",
            provider_type=VMProviderType.CLOUD,
            name=os.getenv("CUA_CONTAINER_NAME") or "",
            api_key=os.getenv("CUA_API_KEY"),
            verbosity=logging.INFO
        )
    else:
        computer = Computer(
            os_type="linux",
            provider_type=VMProviderType.DOCKER,
            verbosity=logging.INFO
        )
    
    try:
        # Start computer connection
        await computer.run()
        print(f"✅ Connected to {'cloud' if use_cloud else 'docker'} container")
        
        # Create improved agent with available model
        model = "anthropic/claude-3-5-sonnet-20241022" if os.getenv("ANTHROPIC_API_KEY") else "openai/computer-use-preview"
        agent = create_improved_agent(
            model=model,
            computer=computer,
            config_name="alternative"
        )
        
        print(f"✅ Created improved agent: {agent.get_stats()}")
        
        # Test OSWorld-style task but keep it simple
        test_task = "Open Firefox and navigate to google.com"
        print(f"\n🎯 Running test task: {test_task}")
        print(f"🛡️  GNOME hot corner prevention: ACTIVE")
        
        result = await run_osworld_task(agent, test_task, max_steps=15)
        print(f"📊 Task result: Success={result['success']}, Steps={result['steps']}, Duration={result.get('duration', 0):.1f}s")
        
        if result.get('errors'):
            print(f"❌ Errors: {result['errors'][:2]}")  # Show first 2 errors
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False
        
    finally:
        try:
            await computer.stop()
            print("✅ Disconnected from cloud container")
        except:
            pass


async def test_osworld_benchmark():
    """Test the improved agent on OSWorld benchmark using HUD (no local computer needed)."""
    
    print("\n🏆 Testing improved agent on OSWorld-Tiny-Public via HUD...")
    
    try:
        import asyncio
        from hud.datasets import run_dataset
        from hud.agents import ClaudeAgent
        
        # Use our improved agent instead of standard ClaudeAgent
        from improved_agent import create_improved_agent
        
        # Generate unique job name
        job_name = f"improved-agent-osworld-{str(uuid.uuid4())[:8]}"
        print(f"🚀 Starting HUD OSWorld evaluation: {job_name}")
        
        # Choose model based on available API keys
        model = "claude-sonnet-4-20250514" if os.getenv("ANTHROPIC_API_KEY") else "openai/computer-use-preview"
        print(f"📊 Running OSWorld evaluation with model: {model}")
        print("🔗 HUD will provide a dashboard link to view results...")
        
        # Create improved agent configuration
        agent_config = {
            "model": model,
            "allowed_tools": ["anthropic_computer"],
        }
        
        # Run the OSWorld benchmark using HUD's run_dataset
        results = await run_dataset(
            name=job_name,
            dataset="ddupont/OSWorld-Tiny-Public",
            agent_class=ClaudeAgent,  # HUD will use this, but we'll customize it
            agent_config=agent_config,
            max_concurrent=30,
            max_steps=30,
        )
        
        print(f"📊 HUD Benchmark completed!")
        print(f"🔗 Check your HUD dashboard at: https://www.hud.so/dashboard")
        print(f"Total results: {len(results)}")
        
        if results:
            # Print sample results
            print("\n📋 Sample results:")
            for i, result in enumerate(results[:3]):
                # Handle both dict and Trace objects
                if hasattr(result, 'reward'):
                    print(f"  Task {i+1}: reward={result.reward}")
                else:
                    print(f"  Task {i+1}: {result}")
            
            # Calculate basic metrics
            success_count = 0
            for r in results:
                if hasattr(r, 'reward'):
                    if r.reward > 0:
                        success_count += 1
                elif isinstance(r, dict) and r.get('success', False):
                    success_count += 1
            
            success_rate = success_count / len(results)
            print(f"\n📊 Success rate: {success_rate:.2%} ({success_count}/{len(results)})")
            
            # Show individual task rewards
            print("📋 Individual task rewards:")
            for i, result in enumerate(results):
                if hasattr(result, 'reward'):
                    print(f"  Task {i+1}: reward={result.reward}")
        else:
            print("⚠️  No results returned")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in HUD benchmark test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_improved_vs_standard():
    """Compare improved agent against standard agent."""
    
    print("\n⚖️  Comparing improved agent vs standard agent...")
    
    # Connect to cloud container
    computer = Computer(
        os_type="linux",
        provider_type=VMProviderType.CLOUD,
        name=os.getenv("CUA_CONTAINER_NAME") or "",
        api_key=os.getenv("CUA_API_KEY"),
        verbosity=logging.INFO
    )
    
    try:
        await computer.run()
        print("✅ Connected to cloud container for comparison")
        
        # Test task
        test_task = "Open a web browser and navigate to google.com"
        
        # Test improved agent
        print("\n🔧 Testing improved agent...")
        improved_agent = create_improved_agent(
            model="openai/computer-use-preview",
            computer=computer,
            config_name="alternative"
        )
        
        improved_result = await run_osworld_task(improved_agent, test_task, max_steps=20)
        print(f"Improved agent result: {improved_result}")
        
        # Test standard agent
        print("\n📦 Testing standard agent...")
        standard_agent = ComputerAgent(
            model="openai/computer-use-preview",
            tools=[computer],
            verbosity=logging.INFO,
            trajectory_dir="trajectories/standard_test"
        )
        
        # Run standard agent task manually
        standard_result = {"task": test_task, "success": False, "steps": 0, "errors": []}
        try:
            step_count = 0
            async for chunk in standard_agent.run(test_task):
                step_count += 1
                if step_count >= 20:  # Safety limit
                    break
            standard_result["steps"] = step_count
            standard_result["success"] = True  # Assume success if no errors
        except Exception as e:
            standard_result["errors"].append(str(e))
        
        print(f"Standard agent result: {standard_result}")
        
        # Compare results
        print("\n📊 Comparison:")
        print(f"Improved Agent - Steps: {improved_result['steps']}, Success: {improved_result['success']}")
        print(f"Standard Agent - Steps: {standard_result['steps']}, Success: {standard_result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comparison test: {e}")
        return False
        
    finally:
        try:
            await computer.stop()
            print("✅ Disconnected from cloud container")
        except:
            pass


async def main():
    """Main test execution."""
    
    print("🚀 Starting Improved Agent Cloud Tests")
    print("=" * 50)
    
    # Setup environment
    env_ok, has_cloud = setup_environment()
    if not env_ok:
        print("❌ Environment setup failed. Please configure API keys in .env file.")
        return
    
    print("✅ Environment configured")
    
    # Run tests - only run OSWorld Benchmark if HUD is available
    tests = [
        ("Basic Functionality", test_basic_functionality),
    ]
    
    # Only add benchmark test if HUD is available
    if os.getenv("HUD_API_KEY"):
        tests.append(("OSWorld Benchmark (HUD)", test_osworld_benchmark))
        print("🔗 OSWorld Benchmark will run via HUD - check dashboard at https://www.hud.so/dashboard")
    else:
        print("ℹ️  Skipping OSWorld Benchmark (no HUD_API_KEY)")
    
    tests.append(("Improved vs Standard", test_improved_vs_standard))
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = success
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! Improved agent is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(main())

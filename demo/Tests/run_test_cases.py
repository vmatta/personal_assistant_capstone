#!/usr/bin/env python
"""Run all 5 test cases and capture outputs with traces for README documentation."""

import json
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO
import time

from src.harness.runner import run_turn
from src.tools.memory import warm_up_async

# Test cases from DEMO_PROMPTS.md
TEST_CASES = [
    {
        "name": "1. ⚡ SIMPLE TASK — Fast Path Reasoning",
        "prompt": "Add a to-do to buy groceries tomorrow at 10 AM",
        "description": "Shows direct reasoning, single agent dispatch, and fast response (4-5 seconds).",
    },
    {
        "name": "2. 🧠 COMPLEX TASK — Tree-of-Thought",
        "prompt": "Schedule beach volleyball Saturday 3 PM in Miami and check the weather for that day",
        "description": "Shows complex reasoning detection, multiple candidate branches, multi-agent dispatch, and real weather data.",
    },
    {
        "name": "3. 💾 MULTI-AGENT WITH NOTES — Preference Saving & Retrieval",
        "prompt": "Remember that I prefer working out in the morning, then schedule a gym session for tomorrow morning at 7 AM",
        "description": "Shows notes agent saving preferences, scheduler creating events, multi-agent orchestration, and preference awareness.",
    },
    {
        "name": "4. 🚨 CONFLICT DETECTION & PREFERENCE GUARDRAIL — Safety",
        "prompt": "Schedule a hiking trip for Sunday afternoon",
        "description": "Shows preference detection, clarification questions, guardrail in action, and smart scheduling.",
    },
    {
        "name": "5. ⏳ SELF-CORRECTION & QA GATE — Reliability",
        "prompt": "Add three tasks: buy milk tomorrow 9 AM, gym Friday 6 PM, call dentist next Wednesday",
        "description": "Shows multiple items handling, QA Critic scoring, automatic retry triggers, and self-correction before returning to user.",
    },
]

def run_all_tests():
    """Run all test cases and return results."""
    print("=" * 80)
    print("PERSONAL ASSISTANT CAPSTONE - TEST CASE EXECUTION")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    # Warm up async resources
    print("Warming up resources (ChromaDB, embeddings)...")
    warm_up_async()
    time.sleep(2)  # Give async tasks time to initialize
    print("✓ Resources warmed up\n")
    
    results = []
    thread_id = None
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}/{len(TEST_CASES)}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"Description: {test_case['description']}\n")
        
        user_input = test_case['prompt']
        print(f"📝 USER INPUT:\n{user_input}\n")
        
        try:
            start_time = time.time()
            print("⏳ Executing...")
            
            # Capture output
            result = run_turn(user_input, thread_id=thread_id)
            execution_time = time.time() - start_time
            
            thread_id = result["thread_id"]
            
            # Extract response
            response = result.get('response', 'No response')
            trace = result.get('trace', 'No trace available')
            reasoning = result.get('reasoning', 'No reasoning available')
            
            print(f"\n✅ FINAL RESPONSE:\n{response}\n")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds\n")
            
            if trace:
                print(f"📊 TRACE:\n{trace}\n")
            
            if reasoning:
                print(f"🧠 REASONING:\n{reasoning}\n")
            
            results.append({
                "test_case_number": i,
                "name": test_case['name'],
                "description": test_case['description'],
                "user_input": user_input,
                "response": response,
                "execution_time": round(execution_time, 2),
                "trace": trace,
                "reasoning": reasoning,
                "status": "✅ PASSED",
            })
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            results.append({
                "test_case_number": i,
                "name": test_case['name'],
                "description": test_case['description'],
                "user_input": user_input,
                "response": f"ERROR: {str(e)}",
                "execution_time": 0,
                "trace": "",
                "reasoning": traceback.format_exc(),
                "status": "❌ FAILED",
            })
        
        print(f"\n{'=' * 80}")
    
    return results


def save_results_to_markdown(results):
    """Save results to markdown file for README."""
    output_path = Path("data/TEST_RESULTS_README.md")
    
    md_content = """# Test Case Results with Traces

This document contains the execution results of all 5 key system test cases, suitable for README documentation.

**Generated:** {timestamp}
**Test Summary:** {passed}/{total} tests passed

---

""".format(
        timestamp=datetime.now().isoformat(),
        passed=sum(1 for r in results if "PASSED" in r["status"]),
        total=len(results),
    )
    
    for result in results:
        md_content += f"""## Test Case {result['test_case_number']}: {result['name']}

**Status:** {result['status']}

### Description
{result['description']}

### User Input
```
{result['user_input']}
```

### Execution Time
{result['execution_time']} seconds

### Response
```
{result['response']}
```

"""
        
        if result['trace']:
            md_content += f"""### Trace
```
{result['trace']}
```

"""
        
        if result['reasoning'] and "ERROR" not in result['reasoning']:
            md_content += f"""### Reasoning
```
{result['reasoning']}
```

"""
        
        md_content += "\n---\n\n"
    
    # Write markdown with UTF-8 encoding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"\n✅ Markdown results saved to: {output_path}")
    
    # Also save JSON for programmatic access with UTF-8 encoding
    json_path = output_path.with_stem("TEST_RESULTS_README").with_suffix(".json")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"✅ JSON results saved to: {json_path}")


def main():
    """Main entry point."""
    try:
        results = run_all_tests()
        save_results_to_markdown(results)
        
        print(f"\n{'=' * 80}")
        print("TEST SUMMARY")
        print(f"{'=' * 80}")
        for result in results:
            print(f"{result['status']} - Test {result['test_case_number']}: {result['name']}")
        
        passed = sum(1 for r in results if "PASSED" in r["status"])
        print(f"\nTotal: {passed}/{len(results)} tests passed")
        print(f"{'=' * 80}\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

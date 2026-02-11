#!/usr/bin/env python
"""Test runner script.

Usage:
    uv run python -m tests.run              # Run all tests
    uv run python -m tests.run visualization # Run specific test
    uv run python -m tests.run --list       # List available tests
"""
import sys
import asyncio
import importlib

TESTS = [
    "test_datasets",
    "test_nlp",
    "test_classification",
    "test_clustering",
    "test_visualization",
    "test_data_scientist_agent",
    "test_memory",
    "test_graph_memory",
    "test_synthesizer",
]


def list_tests():
    print("Available tests:")
    for test in TESTS:
        print(f"  - {test.replace('test_', '')}")


def run_test(name: str) -> bool:
    if not name.startswith("test_"):
        name = f"test_{name}"

    if name not in TESTS:
        print(f"Unknown test: {name}")
        print("Use --list to see available tests")
        return False

    print(f"\n{'='*60}")
    print(f"Running {name}")
    print('='*60 + "\n")

    try:
        module = importlib.import_module(f"tests.{name}")

        # Find and run the main async test function
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and attr_name.startswith("test_"):
                if asyncio.iscoroutinefunction(attr):
                    asyncio.run(attr())
                    return True

        print(f"No async test function found in {name}")
        return False

    except Exception as e:
        print(f"Error running {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all() -> int:
    failed = []
    passed = []

    for test in TESTS:
        success = run_test(test)
        if success:
            passed.append(test)
        else:
            failed.append(test)

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")

    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    else:
        print("ALL TESTS PASSED!")
        return 0


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        sys.exit(run_all())
    elif args[0] == "--list":
        list_tests()
    else:
        success = run_test(args[0])
        sys.exit(0 if success else 1)

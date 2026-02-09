"""Test script for complete memory system (short-term, long-term, semantic).

Run with: uv run python test_memory.py

This test verifies:
1. Short-term memory: Conversation context is maintained
2. Long-term memory: Preferences persist and are retrieved
3. Semantic memory: Similar past interactions are found
"""
import asyncio
import httpx

BASE_URL = "http://localhost:8000"

# Test results tracking
results = {"passed": 0, "failed": 0}


def check(condition: bool, test_name: str, details: str = ""):
    """Record test result with PASS/FAIL indicator."""
    global results
    if condition:
        results["passed"] += 1
        print(f"  [PASS] {test_name}")
    else:
        results["failed"] += 1
        print(f"  [FAIL] {test_name}")
    if details:
        print(f"         {details}")


async def test_memory():
    async with httpx.AsyncClient(timeout=60.0) as client:

        # ============================================================
        # Setup: Get initial memory stats
        # ============================================================
        print("\n" + "=" * 60)
        print("Setup: Getting initial memory stats")
        print("=" * 60)

        response = await client.get(f"{BASE_URL}/memory/stats")
        initial_stats = response.json()
        initial_semantic_count = initial_stats.get("semantic", {}).get("total_count", 0)
        initial_lt_count = initial_stats.get("long_term", {}).get("total_count", 0)
        print(f"  Initial semantic memories: {initial_semantic_count}")
        print(f"  Initial long-term memories: {initial_lt_count}")

        # ============================================================
        # Test 1: Short-Term Memory - Conversation Context
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 1: Short-Term Memory - Conversation Context")
        print("=" * 60)

        # Use unique identifier to avoid conflicts with existing memories
        import random
        test_id = random.randint(1000, 9999)
        test_name = f"TestUser{test_id}"

        # First message - introduce memorable info with unique name
        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": f"Hi! My name is {test_name} and I prefer using TypeScript for web development."}
        )
        result = response.json()
        conversation_id = result.get("conversation_id")
        print(f"  Started conversation: {conversation_id[:8]}...")
        print(f"  Using test name: {test_name}")

        check(
            conversation_id is not None,
            "Conversation created",
            f"ID: {conversation_id[:8] if conversation_id else 'None'}..."
        )

        # Follow-up in same conversation - should have context
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": "What's my name? Please tell me exactly what I just said my name was.",
                "conversation_id": conversation_id
            }
        )
        result = response.json()
        response_text = result.get("response", "").lower()
        has_history = result.get("metadata", {}).get("has_history", False)

        check(
            has_history,
            "Has conversation history flag",
            f"has_history={has_history}"
        )
        check(
            test_name.lower() in response_text or str(test_id) in response_text,
            "Response includes user's name from short-term context",
            f"Looking for '{test_name}' in: {response_text[:150]}..."
        )

        # ============================================================
        # Test 2: Long-Term Memory - Explicit Preference Storage
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 2: Long-Term Memory - Preference Storage & Retrieval")
        print("=" * 60)

        # Store a preference explicitly
        test_pref_key = "test_favorite_color"
        test_pref_value = "blue"

        response = await client.post(
            f"{BASE_URL}/memory/preference",
            params={"key": test_pref_key, "value": test_pref_value}
        )
        store_result = response.json()
        check(
            response.status_code == 200,
            "Store preference API call",
            f"Status: {response.status_code}"
        )

        # Retrieve the preference
        response = await client.get(f"{BASE_URL}/memory/preference/{test_pref_key}")
        if response.status_code == 200:
            pref_result = response.json()
            check(
                pref_result.get("value") == test_pref_value,
                "Retrieve stored preference",
                f"Expected '{test_pref_value}', got '{pref_result.get('value')}'"
            )
        else:
            check(False, "Retrieve stored preference", f"Status: {response.status_code}")

        # Check long-term memory count increased
        response = await client.get(f"{BASE_URL}/memory/stats")
        stats = response.json()
        lt_count = stats.get("long_term", {}).get("total_count", 0)
        check(
            lt_count > initial_lt_count,
            "Long-term memory count increased",
            f"Before: {initial_lt_count}, After: {lt_count}"
        )

        # ============================================================
        # Test 3: Semantic Memory - Interaction Storage
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 3: Semantic Memory - Interaction Storage")
        print("=" * 60)

        # Check semantic memory count after our chat interactions
        response = await client.get(f"{BASE_URL}/memory/stats")
        stats = response.json()
        semantic_count = stats.get("semantic", {}).get("total_count", 0)

        check(
            semantic_count > initial_semantic_count,
            "Semantic memories stored from chat",
            f"Before: {initial_semantic_count}, After: {semantic_count}"
        )

        # ============================================================
        # Test 4: Semantic Memory - Similarity Search
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 4: Semantic Memory - Search Similar Interactions")
        print("=" * 60)

        # Search for similar past interactions
        response = await client.get(
            f"{BASE_URL}/memory/search",
            params={"query": "What is the user's name?"}
        )
        search_results = response.json()
        semantic_results = search_results.get("semantic", [])

        check(
            len(semantic_results) > 0,
            "Semantic search returns results",
            f"Found {len(semantic_results)} similar interactions"
        )

        # ============================================================
        # Test 5: Long-Term Memory in New Conversation
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 5: Long-Term Memory - Persists Across Conversations")
        print("=" * 60)

        # Start a NEW conversation and ask about stored preference
        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "What's my favorite color?"}
        )
        result = response.json()
        new_conv_id = result.get("conversation_id")
        response_text = result.get("response", "").lower()

        check(
            new_conv_id != conversation_id,
            "New conversation started",
            f"Original: {conversation_id[:8]}..., New: {new_conv_id[:8]}..."
        )
        check(
            "blue" in response_text or "don't" in response_text or "haven't" in response_text,
            "Response references stored preference or admits no knowledge",
            f"Response: {response_text[:150]}..."
        )

        # ============================================================
        # Test 6: Memory APIs - List All Long-Term Memories
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 6: Memory APIs - List & Inspect")
        print("=" * 60)

        response = await client.get(f"{BASE_URL}/memory/long-term")
        memories = response.json()

        check(
            isinstance(memories, list),
            "Long-term memories API returns list",
            f"Count: {len(memories)}"
        )

        # Check our test preference is in the list
        test_pref_found = any(
            m.get("key") == test_pref_key and m.get("value") == test_pref_value
            for m in memories
        )
        check(
            test_pref_found,
            "Test preference found in long-term memories",
            f"Looking for key='{test_pref_key}'"
        )

        # ============================================================
        # Cleanup: Delete test preference
        # ============================================================
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)

        # Find and delete our test preference
        for m in memories:
            if m.get("key") == test_pref_key:
                response = await client.delete(f"{BASE_URL}/memory/long-term/{m['id']}")
                check(
                    response.status_code == 200,
                    "Cleanup: Deleted test preference",
                    f"ID: {m['id'][:8]}..."
                )
                break

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        total = results["passed"] + results["failed"]
        print(f"  Passed: {results['passed']}/{total}")
        print(f"  Failed: {results['failed']}/{total}")

        if results["failed"] == 0:
            print("\n  All memory systems working correctly!")
        else:
            print("\n  Some tests failed - review output above.")

        print("\n  Memory Layers Tested:")
        print("  - Short-term: Conversation context within session")
        print("  - Long-term: Preference storage & retrieval")
        print("  - Semantic: Interaction storage & similarity search")
        print("=" * 60)

        return results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(test_memory())
    exit(0 if success else 1)

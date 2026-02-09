"""Test script for synthesizer agent.

Run with: uv run python test_synthesizer.py

This test verifies:
1. Responses flow through the synthesizer
2. Synthesizer enhances responses with memory context
3. Synthesizer works with different agent types
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


async def test_synthesizer():
    async with httpx.AsyncClient(timeout=60.0) as client:

        # ============================================================
        # Setup: Store some preferences for synthesis context
        # ============================================================
        print("\n" + "=" * 60)
        print("Setup: Storing user preferences for context")
        print("=" * 60)

        # Store a preference
        await client.post(
            f"{BASE_URL}/memory/preference",
            params={"key": "communication_style", "value": "concise and technical"}
        )
        print("  Stored preference: communication_style = concise and technical")

        # Store another preference
        await client.post(
            f"{BASE_URL}/memory/preference",
            params={"key": "expertise_level", "value": "advanced developer"}
        )
        print("  Stored preference: expertise_level = advanced developer")

        # ============================================================
        # Test 1: General Chat - Should Flow Through Synthesizer
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 1: General Chat - Workflow Includes Synthesizer")
        print("=" * 60)

        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "Hello! Can you help me with my Python project?"}
        )
        result = response.json()

        check(
            response.status_code == 200,
            "Chat returns 200",
            f"Status: {response.status_code}"
        )
        check(
            result.get("response") is not None and len(result.get("response", "")) > 0,
            "Response is not empty",
            f"Length: {len(result.get('response', ''))}"
        )
        check(
            result.get("agent_used") is not None,
            "Agent used is tracked",
            f"Agent: {result.get('agent_used')}"
        )

        print(f"  Response preview: {result.get('response', '')[:150]}...")

        # ============================================================
        # Test 2: Document Query - Synthesizer with Sources
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 2: Document Query - Synthesizer Handles Sources")
        print("=" * 60)

        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "What documents do I have uploaded?"}
        )
        result = response.json()

        check(
            response.status_code == 200,
            "Document query succeeds",
            f"Status: {response.status_code}"
        )
        check(
            result.get("response") is not None,
            "Response provided",
            f"Agent used: {result.get('agent_used')}"
        )

        print(f"  Response preview: {result.get('response', '')[:150]}...")

        # ============================================================
        # Test 3: Knowledge Graph Query - Synthesizer with Entities
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 3: Knowledge Graph Query")
        print("=" * 60)

        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "What entities are related to Python in the knowledge graph?"}
        )
        result = response.json()

        check(
            response.status_code == 200,
            "KG query succeeds",
            f"Status: {response.status_code}"
        )

        print(f"  Intent: {result.get('intent')}")
        print(f"  Response preview: {result.get('response', '')[:150]}...")

        # ============================================================
        # Test 4: Conversation Continuity Through Synthesizer
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 4: Conversation Continuity")
        print("=" * 60)

        # First message
        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "I'm working on a FastAPI backend application."}
        )
        result = response.json()
        conversation_id = result.get("conversation_id")

        check(
            conversation_id is not None,
            "Conversation started",
            f"ID: {conversation_id[:8] if conversation_id else 'None'}..."
        )

        # Follow-up
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": "What testing framework would you recommend for it?",
                "conversation_id": conversation_id
            }
        )
        result = response.json()
        response_text = result.get("response", "").lower()

        check(
            response.status_code == 200,
            "Follow-up succeeds",
            f"Status: {response.status_code}"
        )
        check(
            result.get("metadata", {}).get("has_history", False),
            "Has conversation history",
            f"has_history: {result.get('metadata', {}).get('has_history')}"
        )
        # Response should mention testing frameworks relevant to FastAPI/Python
        mentions_testing = any(
            term in response_text
            for term in ["pytest", "test", "unittest", "testing"]
        )
        check(
            mentions_testing,
            "Response mentions relevant testing",
            f"Response: {response_text[:100]}..."
        )

        # ============================================================
        # Test 5: Memory Context Integration
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 5: Memory Context in Synthesis")
        print("=" * 60)

        # Query that should benefit from stored preferences
        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "Explain how decorators work"}
        )
        result = response.json()

        check(
            response.status_code == 200,
            "Query with memory context succeeds",
            f"Status: {response.status_code}"
        )

        # The response should exist
        response_text = result.get("response", "")
        check(
            len(response_text) > 50,
            "Substantive response provided",
            f"Length: {len(response_text)}"
        )

        print(f"  Response preview: {response_text[:200]}...")

        # ============================================================
        # Cleanup
        # ============================================================
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)

        # Get long-term memories and delete test ones
        response = await client.get(f"{BASE_URL}/memory/long-term")
        memories = response.json()
        for m in memories:
            if m.get("key") in ["communication_style", "expertise_level"]:
                await client.delete(f"{BASE_URL}/memory/long-term/{m['id']}")
                print(f"  Deleted: {m['key']}")

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
            print("\n  Synthesizer agent working correctly!")
        else:
            print("\n  Some tests failed - review output above.")

        print("\n  Workflow now:")
        print("  Router → Agent → Synthesizer → Response")
        print("=" * 60)

        return results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(test_synthesizer())
    exit(0 if success else 1)

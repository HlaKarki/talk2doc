"""Test script for graph memory system.

Run with: uv run python test_graph_memory.py

This test verifies:
1. Concepts are extracted from conversations
2. Relationships are created between concepts
3. Graph traversal finds related concepts
4. Graph memory is included in agent context
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


async def test_graph_memory():
    async with httpx.AsyncClient(timeout=60.0) as client:

        # ============================================================
        # Setup: Get initial graph stats
        # ============================================================
        print("\n" + "=" * 60)
        print("Setup: Getting initial graph memory stats")
        print("=" * 60)

        response = await client.get(f"{BASE_URL}/memory/stats")
        initial_stats = response.json()
        initial_nodes = initial_stats.get("graph", {}).get("total_nodes", 0)
        initial_edges = initial_stats.get("graph", {}).get("total_edges", 0)
        print(f"  Initial graph nodes: {initial_nodes}")
        print(f"  Initial graph edges: {initial_edges}")

        # ============================================================
        # Test 1: Explicitly Learn Concepts
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 1: Explicitly Learn Concepts via API")
        print("=" * 60)

        # Learn a concept with relationships
        response = await client.post(
            f"{BASE_URL}/memory/learn",
            json={
                "concept": "fastapi",
                "concept_type": "tool",
                "related_to": ["python", "web development", "rest api"]
            }
        )
        learn_result = response.json()
        check(
            response.status_code == 200,
            "Learn concept API call",
            f"Status: {response.status_code}"
        )
        check(
            learn_result.get("edges_created", 0) >= 2,
            "Relationships created",
            f"Edges created: {learn_result.get('edges_created')}"
        )

        # ============================================================
        # Test 2: Get Concept Graph
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 2: Get Concept and Relationships")
        print("=" * 60)

        response = await client.get(f"{BASE_URL}/memory/graph/concept/fastapi")
        if response.status_code == 200:
            graph_result = response.json()
            check(
                graph_result.get("node", {}).get("concept") == "fastapi",
                "Concept retrieved correctly",
                f"Concept: {graph_result.get('node', {}).get('concept')}"
            )
            check(
                len(graph_result.get("related_nodes", [])) > 0,
                "Related nodes found",
                f"Related nodes: {len(graph_result.get('related_nodes', []))}"
            )
            check(
                len(graph_result.get("edges", [])) > 0,
                "Edges retrieved",
                f"Edges: {len(graph_result.get('edges', []))}"
            )
        else:
            check(False, "Get concept graph", f"Status: {response.status_code}")

        # ============================================================
        # Test 3: Search Concepts Semantically
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 3: Semantic Search for Concepts")
        print("=" * 60)

        response = await client.get(
            f"{BASE_URL}/memory/graph/search",
            params={"query": "web framework for building APIs"}
        )
        search_result = response.json()
        check(
            response.status_code == 200,
            "Search API call",
            f"Status: {response.status_code}"
        )
        concepts_found = search_result.get("concepts", [])
        check(
            len(concepts_found) > 0,
            "Concepts found via semantic search",
            f"Found: {[c.get('concept') for c in concepts_found[:3]]}"
        )

        # ============================================================
        # Test 4: Chat and Check Concept Extraction
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 4: Chat - Concepts Extracted Automatically")
        print("=" * 60)

        # Have a conversation mentioning specific concepts
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": "I'm building a machine learning pipeline using scikit-learn and pandas for data preprocessing."
            }
        )
        chat_result = response.json()
        conv_id = chat_result.get('conversation_id', '')
        check(
            response.status_code == 200 and conv_id,
            "Chat API call",
            f"Status: {response.status_code}, Conversation: {conv_id[:8] if conv_id else 'None'}..."
        )
        if response.status_code != 200:
            print(f"         Error: {chat_result}")

        # Wait a moment for background processing
        await asyncio.sleep(1)

        # Check if new concepts were added
        response = await client.get(f"{BASE_URL}/memory/stats")
        new_stats = response.json()
        new_nodes = new_stats.get("graph", {}).get("total_nodes", 0)
        new_edges = new_stats.get("graph", {}).get("total_edges", 0)

        check(
            new_nodes > initial_nodes,
            "New concepts extracted from chat",
            f"Before: {initial_nodes}, After: {new_nodes}"
        )
        print(f"         Nodes by type: {new_stats.get('graph', {}).get('nodes_by_type', {})}")

        # ============================================================
        # Test 5: Get Full Memory Graph
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 5: Get Full Memory Graph")
        print("=" * 60)

        response = await client.get(f"{BASE_URL}/memory/graph")
        graph = response.json()
        check(
            response.status_code == 200,
            "Get full graph API call",
            f"Status: {response.status_code}"
        )
        check(
            len(graph.get("nodes", [])) > 0,
            "Graph has nodes",
            f"Total nodes: {len(graph.get('nodes', []))}"
        )
        check(
            len(graph.get("edges", [])) > 0,
            "Graph has edges",
            f"Total edges: {len(graph.get('edges', []))}"
        )

        # Print some concepts for visibility
        print("         Sample concepts:")
        for node in graph.get("nodes", [])[:5]:
            print(f"           - {node['concept']} ({node['concept_type']})")

        # ============================================================
        # Test 6: Graph Memory in New Chat Context
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 6: Graph Memory Enriches Chat Context")
        print("=" * 60)

        # Ask about a related topic - should have graph context
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": "What tools should I use for data analysis in Python?"
            }
        )
        chat_result = response.json()
        response_text = chat_result.get("response", "").lower()

        check(
            response.status_code == 200 and len(response_text) > 0,
            "Context-aware chat",
            f"Status: {response.status_code}, Response length: {len(response_text)}"
        )
        if response.status_code != 200 or len(response_text) == 0:
            print(f"         Full response: {chat_result}")
        # The response should mention tools we've discussed
        mentions_related = any(
            term in response_text
            for term in ["pandas", "scikit", "sklearn", "machine learning", "data"]
        )
        check(
            mentions_related,
            "Response references related concepts",
            f"Response: {response_text[:150]}..."
        )

        # ============================================================
        # Test 7: Forget a Concept
        # ============================================================
        print("\n" + "=" * 60)
        print("Test 7: Forget a Concept")
        print("=" * 60)

        # Delete the test concept we created
        response = await client.delete(f"{BASE_URL}/memory/graph/concept/fastapi")
        check(
            response.status_code == 200,
            "Delete concept",
            f"Status: {response.status_code}"
        )

        # Verify it's gone
        response = await client.get(f"{BASE_URL}/memory/graph/concept/fastapi")
        check(
            response.status_code == 404,
            "Concept no longer exists",
            f"Status: {response.status_code}"
        )

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
            print("\n  Graph memory system working correctly!")
        else:
            print("\n  Some tests failed - review output above.")

        # Final stats
        response = await client.get(f"{BASE_URL}/memory/stats")
        final_stats = response.json()
        print("\n  Final Graph Memory Stats:")
        print(f"  - Total nodes: {final_stats.get('graph', {}).get('total_nodes', 0)}")
        print(f"  - Total edges: {final_stats.get('graph', {}).get('total_edges', 0)}")
        print(f"  - Nodes by type: {final_stats.get('graph', {}).get('nodes_by_type', {})}")
        print("=" * 60)

        return results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(test_graph_memory())
    exit(0 if success else 1)

"""Test script for memory system."""
import asyncio
import httpx

BASE_URL = "http://localhost:8000"


async def test_memory():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test 1: First message - introduce ourselves
        print("=" * 50)
        print("Test 1: First message (new conversation)")
        print("=" * 50)

        response = await client.post(
            f"{BASE_URL}/chat",
            json={"query": "Hello! My name is Alex and I am learning about RAG systems."}
        )
        result = response.json()
        print(f"Response: {result.get('response', '')[:200]}...")
        print(f"Conversation ID: {result.get('conversation_id')}")
        print(f"Agent used: {result.get('agent_used')}")

        conversation_id = result.get("conversation_id")

        # Test 2: Follow-up message with context
        print("\n" + "=" * 50)
        print("Test 2: Follow-up message (should remember name)")
        print("=" * 50)

        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": "What was my name again? And what am I learning about?",
                "conversation_id": conversation_id
            }
        )
        result = response.json()
        print(f"Response: {result.get('response', '')[:300]}...")
        print(f"Has history: {result.get('metadata', {}).get('has_history')}")

        # Test 3: Check conversations list
        print("\n" + "=" * 50)
        print("Test 3: List conversations")
        print("=" * 50)

        response = await client.get(f"{BASE_URL}/conversations")
        conversations = response.json()
        print(f"Total conversations: {len(conversations)}")
        for conv in conversations[:3]:
            print(f"  - {conv['id'][:8]}... | {conv['title']} | Messages: {conv['message_count']}")

        # Test 4: Get conversation details
        if conversation_id:
            print("\n" + "=" * 50)
            print("Test 4: Get conversation details")
            print("=" * 50)

            response = await client.get(f"{BASE_URL}/conversations/{conversation_id}")
            conv_detail = response.json()
            print(f"Title: {conv_detail['title']}")
            print(f"Messages: {len(conv_detail['messages'])}")
            for msg in conv_detail["messages"]:
                content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"  [{msg['role']}]: {content_preview}")

        print("\n" + "=" * 50)
        print("Memory system test complete!")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_memory())

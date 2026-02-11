"""Test script for Data Scientist Agent."""
import asyncio
import io
import pandas as pd
import numpy as np

from httpx import AsyncClient, ASGITransport
from main import app


async def test_data_scientist_agent():
    """Test the Data Scientist Agent functionality."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING DATA SCIENTIST AGENT")
        print("=" * 60)

        # Create a sample dataset with various data types
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame({
            "customer_id": range(1, n_samples + 1),
            "age": np.random.randint(18, 70, n_samples),
            "income": np.random.normal(50000, 15000, n_samples).round(2),
            "spending_score": np.random.randint(1, 100, n_samples),
            "category": np.random.choice(["Basic", "Standard", "Premium"], n_samples),
            "review": np.random.choice([
                "Great product, highly recommend!",
                "Good quality for the price",
                "Average experience, nothing special",
                "Not satisfied with the purchase",
                "Excellent service and fast delivery",
                "Poor quality, would not buy again",
                "Decent product, meets expectations",
                "Amazing! Best purchase ever!"
            ], n_samples),
            "churned": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Upload the dataset
        print("\n1. Uploading sample customer dataset...")
        files = {"file": ("customers.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        dataset_id = response.json()["id"]
        print(f"   Dataset ID: {dataset_id}")
        print(f"   Rows: {response.json()['row_count']}, Columns: {response.json()['column_count']}")

        # Test 2: Direct agent test via chat endpoint
        print("\n2. Testing agent via chat endpoint...")
        print("   Query: 'Profile my customer dataset and show key statistics'")

        response = await client.post(
            "/chat/",
            json={
                "message": f"Profile the dataset {dataset_id} and show key statistics",
                "conversation_id": None
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   Agent used: {result.get('metadata', {}).get('agent_used', 'unknown')}")
            print(f"   Response preview: {result.get('response', '')[:200]}...")
        else:
            print(f"   Chat endpoint returned: {response.status_code}")
            # This is okay - we'll test the agent directly

        # Test 3: Test the agent directly
        print("\n3. Testing Data Scientist Agent directly...")

        from database.session import AsyncSessionLocal
        from agents.data_scientist_agent import get_data_scientist_agent

        async with AsyncSessionLocal() as db:
            agent = get_data_scientist_agent()

            # Test profiling
            print("\n   a) Testing dataset profiling...")
            response = await agent.invoke(
                f"Profile the dataset and tell me about the data quality",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")
            if response.content:
                print(f"      Insights preview: {response.content[:300]}...")

            # Test sentiment analysis
            print("\n   b) Testing sentiment analysis...")
            response = await agent.invoke(
                "Analyze the sentiment in the review column",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")
            if "sentiment" in str(response.metadata.get("results", {})):
                sentiment_data = response.metadata.get("results", {}).get("analyses", {})
                for key, val in sentiment_data.items():
                    if "sentiment" in key:
                        print(f"      Sentiment results: {val.get('aggregate', {})}")

            # Test keyword extraction
            print("\n   c) Testing keyword extraction...")
            response = await agent.invoke(
                "Extract the top keywords from customer reviews",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")

            # Test clustering
            print("\n   d) Testing clustering...")
            response = await agent.invoke(
                "Cluster customers based on age, income, and spending_score",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")
            if response.metadata.get("results", {}).get("analyses", {}).get("clustering"):
                cluster_info = response.metadata["results"]["analyses"]["clustering"]
                print(f"      Clusters found: {cluster_info.get('n_clusters', 'N/A')}")
                print(f"      Silhouette score: {cluster_info.get('metrics', {}).get('silhouette_score', 'N/A')}")

            # Test visualization
            print("\n   e) Testing visualization request...")
            response = await agent.invoke(
                "Show me the distribution of income",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")
            if response.metadata.get("visualizations"):
                print(f"      Visualizations generated: {len(response.metadata['visualizations'])}")

            # Test complex multi-step query
            print("\n   f) Testing complex multi-step query...")
            response = await agent.invoke(
                "Analyze customer sentiment, find patterns in spending behavior, and suggest customer segments",
                db,
                context={"dataset_id": dataset_id}
            )
            print(f"      Success: {response.success}")
            print(f"      Tools executed: {response.metadata.get('tools_executed', [])}")
            print(f"      Response length: {len(response.content)} characters")

            await db.commit()

        # Cleanup
        print("\n4. Cleaning up...")
        response = await client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        print("   Dataset deleted!")

        print("\n" + "=" * 60)
        print("DATA SCIENTIST AGENT TESTS COMPLETED!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_data_scientist_agent())

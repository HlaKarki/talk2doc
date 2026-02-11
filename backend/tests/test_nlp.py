"""Test script for NLP analysis functionality."""
import asyncio
import io
import pandas as pd
from httpx import AsyncClient, ASGITransport

from main import app


async def test_nlp():
    """Test NLP analysis on dataset text columns."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING NLP ANALYSIS")
        print("=" * 60)

        # Create a sample dataset with text for NLP analysis
        sample_data = {
            "id": list(range(1, 11)),
            "review": [
                "This product is amazing! I love it so much, best purchase ever.",
                "Terrible quality, broke after one day. Very disappointed.",
                "It's okay, nothing special. Does the job.",
                "Absolutely fantastic! Exceeded all my expectations.",
                "Worst experience ever. Do not buy this garbage.",
                "Pretty good for the price. Would recommend.",
                "Not worth the money. Poor customer service too.",
                "Great product, fast shipping, very happy with it!",
                "Meh, it's average. Could be better.",
                "Outstanding quality and value. Five stars!"
            ],
            "category": ["electronics"] * 5 + ["home"] * 5,
            "rating": [5, 1, 3, 5, 1, 4, 2, 5, 3, 5]
        }
        df = pd.DataFrame(sample_data)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Upload the dataset
        print("\n1. Uploading test dataset with reviews...")
        files = {"file": ("product_reviews.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        dataset_id = response.json()["id"]
        print(f"   ✅ Uploaded dataset: {dataset_id}")

        # Test 2: Sentiment Analysis
        print("\n2. Testing sentiment analysis...")
        response = await client.post(
            f"/datasets/{dataset_id}/analyze/sentiment",
            json={"column": "review"}
        )
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200, f"Sentiment failed: {response.text}"
        sentiment_data = response.json()

        print(f"   Column analyzed: {sentiment_data['column']}")
        print(f"   Rows analyzed: {sentiment_data['analyzed_rows']}/{sentiment_data['total_rows']}")
        print(f"   Aggregate results:")
        agg = sentiment_data['aggregate']
        print(f"     - Average polarity: {agg['average_polarity']}")
        print(f"     - Average subjectivity: {agg['average_subjectivity']}")
        print(f"     - Overall sentiment: {agg['overall_sentiment']}")
        print(f"     - Distribution: {agg['label_distribution']}")
        print(f"   Insights:")
        for insight in sentiment_data['insights']:
            print(f"     - {insight}")
        print("   ✅ Sentiment analysis successful!")

        # Test 3: Sentiment with row-level results
        print("\n3. Testing sentiment with per-row results...")
        response = await client.post(
            f"/datasets/{dataset_id}/analyze/sentiment?include_rows=true",
            json={"column": "review"}
        )
        assert response.status_code == 200
        sentiment_rows = response.json()

        print(f"   Sample row results:")
        for i, row in enumerate(sentiment_rows['row_results'][:3]):
            print(f"     Row {i+1}: polarity={row['polarity']}, label={row['label']}")
        print("   ✅ Row-level sentiment successful!")

        # Test 4: Keyword Extraction
        print("\n4. Testing keyword extraction...")
        response = await client.post(
            f"/datasets/{dataset_id}/analyze/keywords",
            json={"column": "review", "top_n": 10}
        )
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200, f"Keywords failed: {response.text}"
        keywords_data = response.json()

        print(f"   Column analyzed: {keywords_data['column']}")
        print(f"   Rows analyzed: {keywords_data['analyzed_rows']}/{keywords_data['total_rows']}")
        print(f"   Top keywords:")
        for kw in keywords_data['top_keywords'][:5]:
            print(f"     - {kw['keyword']}: {kw['score']:.4f}")
        print(f"   Insights:")
        for insight in keywords_data['insights']:
            print(f"     - {insight}")
        print("   ✅ Keyword extraction successful!")

        # Test 5: Error handling - invalid column
        print("\n5. Testing error handling (invalid column)...")
        response = await client.post(
            f"/datasets/{dataset_id}/analyze/sentiment",
            json={"column": "nonexistent_column"}
        )
        print(f"   Status: {response.status_code}")
        assert response.status_code == 400
        print(f"   Error: {response.json()['detail']}")
        print("   ✅ Error handling works!")

        # Cleanup
        print("\n6. Cleaning up...")
        response = await client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        print("   ✅ Dataset deleted!")

        print("\n" + "=" * 60)
        print("ALL NLP TESTS PASSED! ✅")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_nlp())

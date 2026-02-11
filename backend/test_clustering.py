"""Test script for clustering functionality."""
import asyncio
import io
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from main import app


async def test_clustering():
    """Test clustering pipeline."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING CLUSTERING PIPELINE")
        print("=" * 60)

        # Create a sample dataset with natural clusters (customer segments)
        np.random.seed(42)

        # Generate 3 distinct customer segments
        # Segment 1: Young, low income, high spending
        seg1 = pd.DataFrame({
            "age": np.random.normal(25, 5, 40).astype(int),
            "income": np.random.normal(30000, 5000, 40),
            "spending_score": np.random.normal(80, 10, 40),
        })

        # Segment 2: Middle-aged, high income, medium spending
        seg2 = pd.DataFrame({
            "age": np.random.normal(45, 8, 35).astype(int),
            "income": np.random.normal(90000, 15000, 35),
            "spending_score": np.random.normal(50, 15, 35),
        })

        # Segment 3: Older, medium income, low spending
        seg3 = pd.DataFrame({
            "age": np.random.normal(60, 7, 25).astype(int),
            "income": np.random.normal(55000, 10000, 25),
            "spending_score": np.random.normal(30, 12, 25),
        })

        df = pd.concat([seg1, seg2, seg3], ignore_index=True)
        df["customer_id"] = range(1, len(df) + 1)
        df = df[["customer_id", "age", "income", "spending_score"]]

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Upload the dataset
        print("\n1. Uploading customer segmentation dataset...")
        files = {"file": ("customers.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        dataset_id = response.json()["id"]
        print(f"   ✅ Uploaded dataset: {dataset_id}")
        print(f"   Rows: {response.json()['row_count']}")

        # Test 2: K-Means clustering (auto-detect K)
        print("\n2. Testing K-Means clustering (auto-detect K)...")
        response = await client.post(
            f"/models/cluster/{dataset_id}",
            json={
                "feature_columns": ["age", "income", "spending_score"],
                "algorithm": "kmeans"
            }
        )
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200, f"Clustering failed: {response.text}"
        kmeans_result = response.json()

        print(f"   Algorithm: {kmeans_result['algorithm']}")
        print(f"   Clusters found: {kmeans_result['n_clusters']}")
        print(f"   Metrics:")
        for key, value in kmeans_result['metrics'].items():
            print(f"     - {key}: {value}")
        print(f"   Cluster profiles:")
        for profile in kmeans_result['cluster_profiles']:
            print(f"     - {profile['name']}: {profile['size']} samples ({profile['percentage']}%)")
            stats = profile['feature_stats']
            print(f"       age: mean={stats['age']['mean']:.1f}")
            print(f"       income: mean=${stats['income']['mean']:,.0f}")
            print(f"       spending: mean={stats['spending_score']['mean']:.1f}")
        print(f"   Visualization: {len(kmeans_result['visualization']['x'])} points")
        print(f"   PCA explained variance: {[round(v, 3) for v in kmeans_result['visualization']['explained_variance']]}")
        print("   ✅ K-Means clustering successful!")

        # Test 3: K-Means with specified K
        print("\n3. Testing K-Means with n_clusters=4...")
        response = await client.post(
            f"/models/cluster/{dataset_id}",
            json={
                "feature_columns": ["age", "income", "spending_score"],
                "algorithm": "kmeans",
                "n_clusters": 4
            }
        )
        assert response.status_code == 200
        result = response.json()
        print(f"   Clusters: {result['n_clusters']}")
        print(f"   Silhouette score: {result['metrics'].get('silhouette_score', 'N/A')}")
        print("   ✅ K-Means with specified K successful!")

        # Test 4: Hierarchical clustering
        print("\n4. Testing Hierarchical clustering...")
        response = await client.post(
            f"/models/cluster/{dataset_id}",
            json={
                "feature_columns": ["age", "income", "spending_score"],
                "algorithm": "hierarchical",
                "n_clusters": 3
            }
        )
        assert response.status_code == 200
        hier_result = response.json()
        print(f"   Clusters: {hier_result['n_clusters']}")
        print(f"   Silhouette score: {hier_result['metrics'].get('silhouette_score', 'N/A')}")
        print("   ✅ Hierarchical clustering successful!")

        # Test 5: DBSCAN clustering
        print("\n5. Testing DBSCAN clustering...")
        response = await client.post(
            f"/models/cluster/{dataset_id}",
            json={
                "feature_columns": ["age", "income", "spending_score"],
                "algorithm": "dbscan"
            }
        )
        assert response.status_code == 200
        dbscan_result = response.json()
        print(f"   Clusters: {dbscan_result['n_clusters']}")
        print(f"   Noise points: {dbscan_result['noise_points']}")
        print("   ✅ DBSCAN clustering successful!")

        # Test 6: Error handling - invalid column
        print("\n6. Testing error handling (invalid column)...")
        response = await client.post(
            f"/models/cluster/{dataset_id}",
            json={
                "feature_columns": ["nonexistent_column"],
                "algorithm": "kmeans"
            }
        )
        print(f"   Status: {response.status_code}")
        assert response.status_code == 400
        print(f"   Error: {response.json()['detail']}")
        print("   ✅ Error handling works!")

        # Cleanup
        print("\n7. Cleaning up...")
        response = await client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        print("   ✅ Dataset deleted!")

        print("\n" + "=" * 60)
        print("ALL CLUSTERING TESTS PASSED! ✅")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_clustering())

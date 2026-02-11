"""Test script for dataset upload and profiling functionality."""
import asyncio
import io
import pandas as pd
from httpx import AsyncClient, ASGITransport

from main import app


async def test_datasets():
    """Test dataset upload, profiling, and management."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING DATASET UPLOAD AND PROFILING")
        print("=" * 60)

        # Create a sample CSV in memory
        sample_data = {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "age": [25, 30, 35, 28, 42, 31, 27, 45, 33, 29],
            "salary": [50000, 60000, 75000, 55000, 90000, 65000, 52000, 95000, 70000, 58000],
            "department": ["Engineering", "Sales", "Engineering", "Marketing", "Engineering", "Sales", "Marketing", "Engineering", "Sales", "Marketing"],
            "score": [85.5, 92.0, 78.3, 88.7, 95.2, 82.1, 79.8, 91.5, 86.4, 84.0],
            "active": [True, True, False, True, True, False, True, True, True, False],
        }
        df = pd.DataFrame(sample_data)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Test 1: Upload dataset
        print("\n1. Testing dataset upload...")
        files = {"file": ("test_employees.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

        assert response.status_code == 200, f"Upload failed: {response.text}"
        upload_data = response.json()
        dataset_id = upload_data["id"]
        assert upload_data["status"] == "uploaded"
        assert upload_data["row_count"] == 10
        assert upload_data["column_count"] == 7
        print("   ✅ Upload successful!")

        # Test 2: List datasets
        print("\n2. Testing list datasets...")
        response = await client.get("/datasets")
        print(f"   Status: {response.status_code}")
        list_data = response.json()
        print(f"   Total datasets: {list_data['total']}")

        assert response.status_code == 200
        assert list_data["total"] >= 1
        print("   ✅ List successful!")

        # Test 3: Get dataset details
        print("\n3. Testing get dataset details...")
        response = await client.get(f"/datasets/{dataset_id}")
        print(f"   Status: {response.status_code}")
        dataset_data = response.json()
        print(f"   Filename: {dataset_data['filename']}")
        print(f"   Rows: {dataset_data['row_count']}, Columns: {dataset_data['column_count']}")
        print(f"   Schema columns: {dataset_data['column_schema']['column_names']}")

        assert response.status_code == 200
        assert dataset_data["filename"] == "test_employees.csv"
        print("   ✅ Get details successful!")

        # Test 4: Preview dataset
        print("\n4. Testing dataset preview...")
        response = await client.get(f"/datasets/{dataset_id}/preview?limit=5")
        print(f"   Status: {response.status_code}")
        preview_data = response.json()
        print(f"   Columns: {preview_data['columns']}")
        print(f"   Preview rows: {preview_data['preview_rows']} of {preview_data['total_rows']}")
        print(f"   First row: {preview_data['rows'][0]}")

        assert response.status_code == 200
        assert preview_data["preview_rows"] == 5
        assert preview_data["total_rows"] == 10
        print("   ✅ Preview successful!")

        # Test 5: Get profile
        print("\n5. Testing dataset profiling...")
        response = await client.get(f"/datasets/{dataset_id}/profile")
        print(f"   Status: {response.status_code}")
        profile_data = response.json()

        # Basic stats
        basic = profile_data["basic_stats"]
        print(f"   Basic Stats:")
        print(f"     - Rows: {basic['row_count']}, Columns: {basic['column_count']}")
        print(f"     - Memory: {basic['memory_usage_mb']} MB")
        print(f"     - Duplicates: {basic['duplicate_rows']} ({basic['duplicate_row_percentage']}%)")

        # Column profiles
        print(f"   Column Profiles:")
        for col in profile_data["column_profiles"]:
            print(f"     - {col['name']}: {col['dtype']} ({col['semantic_type']}), missing: {col['missing_percentage']}%")
            if "mean" in col:
                print(f"       mean={col['mean']}, std={col['std']}, min={col['min']}, max={col['max']}")

        # Correlations
        corr = profile_data["correlations"]
        if "high_correlations" in corr:
            print(f"   High Correlations: {len(corr['high_correlations'])} pairs")
            for hc in corr["high_correlations"]:
                print(f"     - {hc['column1']} <-> {hc['column2']}: {hc['correlation']:.3f}")

        # Data quality
        quality = profile_data["data_quality"]
        print(f"   Data Quality Score: {quality['quality_score']}/100")
        print(f"   Issues found: {quality['total_issues']}")
        for issue in quality["issues"]:
            print(f"     - [{issue['severity']}] {issue['message']}")

        assert response.status_code == 200
        assert basic["row_count"] == 10
        print("   ✅ Profiling successful!")

        # Test 6: Regenerate profile
        print("\n6. Testing profile regeneration...")
        response = await client.post(f"/datasets/{dataset_id}/profile")
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200
        print("   ✅ Regeneration successful!")

        # Test 7: Delete dataset
        print("\n7. Testing dataset deletion...")
        response = await client.delete(f"/datasets/{dataset_id}")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")

        assert response.status_code == 200
        print("   ✅ Deletion successful!")

        # Verify deletion
        response = await client.get(f"/datasets/{dataset_id}")
        assert response.status_code == 404
        print("   ✅ Verified dataset is gone!")

        # Test 8: Error handling - invalid file type
        print("\n8. Testing error handling (invalid file type)...")
        txt_buffer = io.BytesIO(b"This is a text file")
        files = {"file": ("test.txt", txt_buffer, "text/plain")}
        response = await client.post("/datasets/upload", files=files)
        print(f"   Status: {response.status_code}")
        print(f"   Error: {response.json()['detail']}")

        assert response.status_code == 400
        print("   ✅ Error handling works!")

        print("\n" + "=" * 60)
        print("ALL DATASET TESTS PASSED! ✅")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_datasets())

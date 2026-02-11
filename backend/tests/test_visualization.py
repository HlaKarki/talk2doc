"""Test script for visualization functionality."""
import asyncio
import io
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from main import app


async def test_visualization():
    """Test visualization pipeline."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING VISUALIZATION PIPELINE")
        print("=" * 60)

        # Create a sample dataset with various data types
        np.random.seed(42)
        n_samples = 100

        # Generate dates
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

        df = pd.DataFrame({
            "date": dates,
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "value": np.random.normal(100, 20, n_samples),
            "quantity": np.random.randint(1, 50, n_samples),
            "price": np.random.uniform(10, 100, n_samples).round(2),
            "rating": np.random.choice([1, 2, 3, 4, 5], n_samples),
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Upload the dataset
        print("\n1. Uploading sample dataset...")
        files = {"file": ("sample_data.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        dataset_id = response.json()["id"]
        print(f"   Dataset ID: {dataset_id}")
        print(f"   Rows: {response.json()['row_count']}, Columns: {response.json()['column_count']}")

        # Test 2: Distribution plot (histogram)
        print("\n2. Testing distribution plot (histogram)...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/distribution",
            json={"column": "value", "plot_type": "histogram"}
        )
        assert response.status_code == 200, f"Distribution failed: {response.text}"
        result = response.json()
        assert result["type"] == "distribution"
        assert "data" in result["figure"]
        assert "layout" in result["figure"]
        print(f"   Type: {result['type']}")
        print(f"   Figure has {len(result['figure']['data'])} trace(s)")
        print(f"   Title: {result['figure']['layout'].get('title', {}).get('text', 'N/A')}")

        # Test 3: Distribution plot (box)
        print("\n3. Testing distribution plot (box)...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/distribution",
            json={"column": "price", "plot_type": "box"}
        )
        assert response.status_code == 200
        result = response.json()
        print(f"   Box plot generated for 'price' column")

        # Test 4: Distribution plot (bar for categorical)
        print("\n4. Testing distribution plot (bar for categorical)...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/distribution",
            json={"column": "category", "plot_type": "auto"}
        )
        assert response.status_code == 200
        result = response.json()
        print(f"   Bar chart generated for 'category' column")

        # Test 5: Scatter plot
        print("\n5. Testing scatter plot...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/scatter",
            json={
                "x_column": "quantity",
                "y_column": "price",
                "color_by": "category"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "scatter"
        print(f"   Scatter plot: quantity vs price, colored by category")
        print(f"   Number of traces: {len(result['figure']['data'])}")

        # Test 6: Correlation matrix
        print("\n6. Testing correlation matrix...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/correlation",
            json={"columns": ["value", "quantity", "price", "rating"]}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "correlation"
        print(f"   Correlation matrix generated for 4 numeric columns")

        # Test 7: Time series plot
        print("\n7. Testing time series plot...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/timeseries",
            json={
                "date_column": "date",
                "value_column": "value",
                "group_by": "category"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "timeseries"
        print(f"   Time series: value over date, grouped by category")
        print(f"   Number of lines: {len(result['figure']['data'])}")

        # Test 8: Pairplot
        print("\n8. Testing pairplot...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/pairplot",
            json={
                "columns": ["value", "quantity", "price"],
                "color_by": "category"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "pairplot"
        print(f"   Pairplot generated for 3 columns")

        # Test 9: Auto visualization
        print("\n9. Testing auto visualization...")
        response = await client.get(
            f"/datasets/{dataset_id}/visualize/auto?max_charts=5"
        )
        assert response.status_code == 200
        result = response.json()
        print(f"   Auto-generated {result['total']} charts:")
        for chart in result["charts"]:
            chart_type = chart.get("type", "unknown")
            column = chart.get("column", chart.get("columns", "multiple"))
            print(f"     - {chart_type}: {column}")

        # Test 10: Error handling - invalid column
        print("\n10. Testing error handling (invalid column)...")
        response = await client.post(
            f"/datasets/{dataset_id}/visualize/distribution",
            json={"column": "nonexistent_column"}
        )
        assert response.status_code == 400
        print(f"   Error correctly returned: {response.json()['detail']}")

        # Test 11: Error handling - correlation with insufficient columns
        print("\n11. Testing error handling (insufficient numeric columns)...")
        # Create dataset with only 1 numeric column
        df_single = pd.DataFrame({
            "name": ["A", "B", "C"],
            "value": [1, 2, 3]
        })
        csv_single = io.BytesIO()
        df_single.to_csv(csv_single, index=False)
        csv_single.seek(0)

        files = {"file": ("single_col.csv", csv_single, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        single_id = response.json()["id"]

        response = await client.post(
            f"/datasets/{single_id}/visualize/correlation",
            json={}
        )
        assert response.status_code == 400
        print(f"   Error correctly returned for single numeric column")

        # Cleanup
        print("\n12. Cleaning up...")
        response = await client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        response = await client.delete(f"/datasets/{single_id}")
        assert response.status_code == 200
        print("   Datasets deleted!")

        print("\n" + "=" * 60)
        print("ALL VISUALIZATION TESTS PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_visualization())

"""Test script for ML classification functionality."""
import asyncio
import io
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport

from main import app


async def test_classification():
    """Test ML classification pipeline."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("=" * 60)
        print("TESTING ML CLASSIFICATION PIPELINE")
        print("=" * 60)

        # Create a sample dataset (simplified Titanic-like data)
        np.random.seed(42)
        n_samples = 100

        sample_data = {
            "passenger_id": list(range(1, n_samples + 1)),
            "pclass": np.random.choice([1, 2, 3], n_samples).tolist(),
            "sex": np.random.choice(["male", "female"], n_samples).tolist(),
            "age": np.random.randint(1, 80, n_samples).tolist(),
            "fare": np.random.uniform(5, 500, n_samples).round(2).tolist(),
            "embarked": np.random.choice(["S", "C", "Q"], n_samples).tolist(),
        }

        # Create target based on features (women and 1st class have higher survival)
        survived = []
        for i in range(n_samples):
            prob = 0.3
            if sample_data["sex"][i] == "female":
                prob += 0.4
            if sample_data["pclass"][i] == 1:
                prob += 0.2
            survived.append(1 if np.random.random() < prob else 0)
        sample_data["survived"] = survived

        df = pd.DataFrame(sample_data)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Upload the dataset
        print("\n1. Uploading Titanic-like dataset...")
        files = {"file": ("titanic_sample.csv", csv_buffer, "text/csv")}
        response = await client.post("/datasets/upload", files=files)
        assert response.status_code == 200, f"Upload failed: {response.text}"
        dataset_id = response.json()["id"]
        print(f"   ✅ Uploaded dataset: {dataset_id}")
        print(f"   Rows: {response.json()['row_count']}, Columns: {response.json()['column_count']}")

        # Test 2: Train Random Forest classifier
        print("\n2. Training Random Forest classifier...")
        response = await client.post(
            f"/models/train/{dataset_id}",
            json={
                "target_column": "survived",
                "feature_columns": ["pclass", "sex", "age", "fare", "embarked"],
                "algorithm": "random_forest"
            }
        )
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200, f"Training failed: {response.text}"
        train_data = response.json()
        model_id = train_data["model_id"]

        print(f"   Model ID: {model_id}")
        print(f"   Algorithm: {train_data['algorithm']}")
        print(f"   Message: {train_data['message']}")
        print(f"   Metrics:")
        for key, value in train_data["metrics"].items():
            print(f"     - {key}: {value}")
        print(f"   Feature Importance:")
        for fi in train_data["feature_importance"][:3]:
            print(f"     - {fi['feature']}: {fi['importance']:.4f}")
        print(f"   Confusion Matrix: {train_data['confusion_matrix']['matrix']}")
        print("   ✅ Random Forest trained successfully!")

        # Test 3: Make predictions
        print("\n3. Making predictions...")
        test_samples = [
            {"pclass": 1, "sex": "female", "age": 25, "fare": 100.0, "embarked": "C"},
            {"pclass": 3, "sex": "male", "age": 35, "fare": 10.0, "embarked": "S"},
            {"pclass": 2, "sex": "female", "age": 45, "fare": 50.0, "embarked": "Q"},
        ]
        response = await client.post(
            f"/models/{model_id}/predict",
            json={"data": test_samples}
        )
        print(f"   Status: {response.status_code}")

        assert response.status_code == 200, f"Prediction failed: {response.text}"
        pred_data = response.json()

        print(f"   Predictions: {pred_data['predictions']}")
        if pred_data["probabilities"]:
            print(f"   Probabilities (first sample): {[round(p, 3) for p in pred_data['probabilities'][0]]}")
        print("   ✅ Predictions successful!")

        # Test 4: Train Logistic Regression
        print("\n4. Training Logistic Regression classifier...")
        response = await client.post(
            f"/models/train/{dataset_id}",
            json={
                "target_column": "survived",
                "feature_columns": ["pclass", "sex", "age", "fare"],
                "algorithm": "logistic_regression"
            }
        )
        assert response.status_code == 200, f"Training failed: {response.text}"
        lr_data = response.json()
        lr_model_id = lr_data["model_id"]

        print(f"   Model ID: {lr_model_id}")
        print(f"   Accuracy: {lr_data['metrics']['accuracy']}")
        print(f"   F1 Score: {lr_data['metrics']['f1_score']}")
        print("   ✅ Logistic Regression trained successfully!")

        # Test 5: List all models
        print("\n5. Listing all models...")
        response = await client.get("/models")
        assert response.status_code == 200
        models_data = response.json()

        print(f"   Total models: {models_data['total']}")
        for m in models_data["models"]:
            print(f"     - {m['algorithm']}: accuracy={m['metrics']['accuracy']}")
        print("   ✅ List models successful!")

        # Test 6: Get specific model
        print("\n6. Getting model details...")
        response = await client.get(f"/models/{model_id}")
        assert response.status_code == 200
        model_detail = response.json()

        print(f"   Algorithm: {model_detail['algorithm']}")
        print(f"   Target: {model_detail['target_column']}")
        print(f"   Features: {model_detail['feature_columns']}")
        print("   ✅ Get model successful!")

        # Test 7: Error handling - missing features in prediction
        print("\n7. Testing error handling (missing features)...")
        response = await client.post(
            f"/models/{model_id}/predict",
            json={"data": [{"pclass": 1, "sex": "female"}]}  # Missing age, fare, embarked
        )
        print(f"   Status: {response.status_code}")
        assert response.status_code == 400
        print(f"   Error: {response.json()['detail']}")
        print("   ✅ Error handling works!")

        # Test 8: Delete models
        print("\n8. Cleaning up models...")
        response = await client.delete(f"/models/{model_id}")
        assert response.status_code == 200
        response = await client.delete(f"/models/{lr_model_id}")
        assert response.status_code == 200
        print("   ✅ Models deleted!")

        # Cleanup dataset
        print("\n9. Cleaning up dataset...")
        response = await client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        print("   ✅ Dataset deleted!")

        print("\n" + "=" * 60)
        print("ALL CLASSIFICATION TESTS PASSED! ✅")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_classification())

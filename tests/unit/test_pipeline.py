"""
test_pipeline.py – Unit tests cho ML Pipeline
Tích hợp từ Lab 3: Testing & CI/CD
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataIngestion:
    """Unit tests cho pipeline/data_ingestion.py"""

    def test_load_and_split_returns_correct_shapes(self):
        from pipeline.data_ingestion import load_and_split
        X_train, X_test, y_train, y_test, feature_names, stats = load_and_split()
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(feature_names) == X_train.shape[1]

    def test_feature_count_is_13(self):
        from pipeline.data_ingestion import load_and_split
        X_train, _, _, _, feature_names, _ = load_and_split()
        assert X_train.shape[1] == 13, "Credit card dataset phải có 13 features"
        assert len(feature_names) == 13

    def test_split_ratio_approximately_80_20(self):
        from pipeline.data_ingestion import load_and_split
        X_train, X_test, y_train, y_test, _, stats = load_and_split(test_size=0.2)
        total = stats["n_train"] + stats["n_test"]
        train_ratio = stats["n_train"] / total
        assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio:.2f} không phải ~80%"

    def test_no_missing_values_in_dataset(self):
        from pipeline.data_ingestion import load_and_split
        X_train, X_test, _, _, _, _ = load_and_split()
        assert not np.isnan(X_train).any(), "Train set không được có NaN"
        assert not np.isnan(X_test).any(), "Test set không được có NaN"

    def test_class_distribution(self):
        from pipeline.data_ingestion import load_and_split
        _, _, y_train, y_test, _, stats = load_and_split()
        assert stats["n_classes"] == 2, "Credit card fraud dataset phải có 2 classes"
        classes_in_train = set(np.unique(y_train))
        classes_in_test = set(np.unique(y_test))
        assert classes_in_train == {0, 1}
        assert classes_in_test == {0, 1}

    def test_stats_dict_contains_required_keys(self):
        from pipeline.data_ingestion import load_and_split
        _, _, _, _, _, stats = load_and_split()
        required_keys = ["n_total", "n_train", "n_test", "n_features", "n_classes"]
        for key in required_keys:
            assert key in stats, f"stats thiếu key '{key}'"


class TestPreprocessing:
    """Unit tests cho pipeline/preprocessing.py"""

    def test_scaler_normalizes_to_unit_std(self, fraud_data):
        from pipeline.preprocessing import preprocess
        X_train_s, _, _, report = preprocess(
            fraud_data["X_train"], fraud_data["X_test"], fraud_data["feature_names"]
        )
        std = X_train_s.std(axis=0)
        assert all(abs(s - 1.0) < 0.01 for s in std), "StandardScaler phải cho std ≈ 1"

    def test_scaler_normalizes_mean_to_zero(self, fraud_data):
        from pipeline.preprocessing import preprocess
        X_train_s, _, _, _ = preprocess(
            fraud_data["X_train"], fraud_data["X_test"], fraud_data["feature_names"]
        )
        mean = X_train_s.mean(axis=0)
        assert all(abs(m) < 0.01 for m in mean), "StandardScaler phải cho mean ≈ 0"

    def test_output_shape_unchanged(self, fraud_data):
        from pipeline.preprocessing import preprocess
        X_train_s, X_test_s, _, _ = preprocess(
            fraud_data["X_train"], fraud_data["X_test"], fraud_data["feature_names"]
        )
        assert X_train_s.shape == fraud_data["X_train"].shape
        assert X_test_s.shape == fraud_data["X_test"].shape

    def test_no_missing_after_scaling(self, fraud_data):
        from pipeline.preprocessing import preprocess
        X_train_s, X_test_s, _, report = preprocess(
            fraud_data["X_train"], fraud_data["X_test"], fraud_data["feature_names"]
        )
        assert report["missing_train"] == 0
        assert report["missing_test"] == 0

    def test_scaler_save_and_load(self, fraud_data, tmp_path):
        from pipeline.preprocessing import preprocess, save_scaler, load_scaler
        _, _, scaler, _ = preprocess(
            fraud_data["X_train"], fraud_data["X_test"], fraud_data["feature_names"]
        )
        save_scaler(scaler, str(tmp_path))
        loaded = load_scaler(str(tmp_path))
        orig = scaler.transform(fraud_data["X_test"])
        loaded_result = loaded.transform(fraud_data["X_test"])
        np.testing.assert_array_almost_equal(orig, loaded_result)


class TestModel:
    """Unit tests cho model (RandomForest)"""

    def test_model_is_fitted(self, trained_model):
        model, _ = trained_model
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(model)  # Không raise = OK

    def test_prediction_classes_are_valid(self, trained_model, fraud_data):
        model, scaler = trained_model
        X_test_s = scaler.transform(fraud_data["X_test"])
        preds = model.predict(X_test_s)
        assert all(p in {0, 1} for p in preds), "Prediction phải là 0 (legitimate) hoặc 1 (fraud)"

    def test_prediction_output_shape(self, trained_model, fraud_data):
        model, scaler = trained_model
        X_test_s = scaler.transform(fraud_data["X_test"])
        preds = model.predict(X_test_s)
        assert len(preds) == len(fraud_data["X_test"])

    def test_model_accuracy_above_threshold(self, trained_model, fraud_data):
        from sklearn.metrics import accuracy_score
        model, scaler = trained_model
        X_test_s = scaler.transform(fraud_data["X_test"])
        preds = model.predict(X_test_s)
        acc = accuracy_score(fraud_data["y_test"], preds)
        assert acc >= 0.75, f"Model accuracy {acc:.4f} quá thấp (< 0.75)"

    def test_predict_proba_sums_to_one(self, trained_model, fraud_data):
        model, scaler = trained_model
        X_test_s = scaler.transform(fraud_data["X_test"][:5])
        proba = model.predict_proba(X_test_s)
        for row in proba:
            assert abs(sum(row) - 1.0) < 1e-6, "Probabilities phải sum = 1"

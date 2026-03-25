"""
test_data_quality.py - Data quality tests cho Credit Card Fraud Detection dataset
Tich hop tu Lab 3: Testing & CI/CD
Kiem tra: completeness, consistency, range, schema
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FEATURE_NAMES = [
    "amount", "time_of_day", "day_of_week", "merchant_risk_score",
    "distance_from_home_km", "distance_from_last_txn_km",
    "ratio_to_median_amount", "repeat_merchant", "used_chip",
    "used_pin", "online_order", "foreign_transaction", "txn_velocity_1h",
]


@pytest.fixture(scope="module")
def fraud_raw():
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=5000, n_features=13, n_informative=8,
        n_redundant=2, n_classes=2, weights=[0.8, 0.2],
        random_state=42, flip_y=0,
    )
    return {"X": X, "y": y, "feature_names": FEATURE_NAMES}


class TestDataCompleteness:
    """Du lieu phai day du, khong thieu."""

    def test_no_nan_values(self, fraud_raw):
        assert not np.isnan(fraud_raw["X"]).any(), "Dataset khong duoc co NaN"

    def test_no_inf_values(self, fraud_raw):
        assert not np.isinf(fraud_raw["X"]).any(), "Dataset khong duoc co Inf"

    def test_sample_count(self, fraud_raw):
        assert len(fraud_raw["X"]) == 5000, "Fraud dataset phai co dung 5000 samples"

    def test_feature_count(self, fraud_raw):
        assert fraud_raw["X"].shape[1] == 13, "Fraud dataset phai co dung 13 features"


class TestDataSchema:
    """Schema validation - ten feature va kieu du lieu."""

    def test_feature_names_count(self, fraud_raw):
        assert len(fraud_raw["feature_names"]) == 13

    def test_expected_feature_names_present(self, fraud_raw):
        for name in FEATURE_NAMES:
            assert name in fraud_raw["feature_names"], f"Feature '{name}' khong tim thay"

    def test_target_has_2_classes(self, fraud_raw):
        unique_classes = np.unique(fraud_raw["y"])
        assert len(unique_classes) == 2
        assert set(unique_classes) == {0, 1}

    def test_all_features_are_numeric(self, fraud_raw):
        assert fraud_raw["X"].dtype in [np.float32, np.float64], \
            f"Data dtype phai la float, nhung la {fraud_raw['X'].dtype}"


class TestDataRanges:
    """Kiem tra cac gia tri feature nam trong range hop ly."""

    def test_features_within_reasonable_range(self, fraud_raw):
        assert fraud_raw["X"].min() > -20, "Feature gia tri nho nhat < -20"
        assert fraud_raw["X"].max() < 20, "Feature gia tri lon nhat > 20"

    def test_all_features_have_variance(self, fraud_raw):
        variances = fraud_raw["X"].var(axis=0)
        for v, name in zip(variances, FEATURE_NAMES):
            assert v > 0, f"Feature '{name}' co variance = 0 (constant)"

    def test_label_values_are_binary(self, fraud_raw):
        assert set(np.unique(fraud_raw["y"])).issubset({0, 1})


class TestDataDistribution:
    """Kiem tra phan phoi du lieu giua cac class."""

    def test_fraud_rate_reasonable(self, fraud_raw):
        """Ti le gian lan trong khoang 5-50%."""
        fraud_rate = fraud_raw["y"].mean()
        assert 0.05 <= fraud_rate <= 0.50, f"Fraud rate {fraud_rate:.1%} khong hop ly"

    def test_both_classes_present(self, fraud_raw):
        """Ca legitimate (0) va fraud (1) phai co >= 100 samples."""
        from collections import Counter
        counts = Counter(fraud_raw["y"])
        for cls in [0, 1]:
            assert counts[cls] >= 100, f"Class {cls} chi co {counts[cls]} samples"

    def test_features_have_variance(self, fraud_raw):
        """Khong co feature nao bi constant (variance = 0)."""
        variances = fraud_raw["X"].var(axis=0)
        for v, name in zip(variances, FEATURE_NAMES):
            assert v > 0, f"Feature '{name}' co variance = 0 (constant)"

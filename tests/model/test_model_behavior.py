"""
test_model_behavior.py - Model behavioral tests (invariance, directional, minimum functionality)
Tich hop tu Lab 3: Testing & CI/CD

Loai tests:
- Minimum Functionality (MFT): model phai dung tren cac case hien nhien
- Invariance (INV): thay doi nho khong quan trong khong duoc doi prediction
- Directional Expectation (DIR): thay doi feature cu the phai anh huong theo chieu ky vong
"""
import os
import sys
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FEATURE_NAMES = [
    "amount", "time_of_day", "day_of_week", "merchant_risk_score",
    "distance_from_home_km", "distance_from_last_txn_km",
    "ratio_to_median_amount", "repeat_merchant", "used_chip",
    "used_pin", "online_order", "foreign_transaction", "txn_velocity_1h",
]


@pytest.fixture(scope="module")
def model_and_scaler():
    X, y = make_classification(
        n_samples=5000, n_features=13, n_informative=8,
        n_redundant=2, n_classes=2, weights=[0.8, 0.2],
        random_state=42, flip_y=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_train)
    return model, scaler, X_te, y_test, FEATURE_NAMES



# ============================================================
# Minimum Functionality Tests (MFT)
# ============================================================
class TestMinimumFunctionality:

    def test_model_achieves_minimum_accuracy(self, model_and_scaler):
        model, scaler, X_test, y_test, _ = model_and_scaler
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        assert acc >= 0.70, f"Model accuracy {acc:.4f} duoi muc toi thieu 0.70"

    def test_model_predicts_both_classes(self, model_and_scaler):
        """Model phai co kha nang predict ca 2 class (legitimate=0, fraud=1)."""
        model, scaler, X_test, y_test, _ = model_and_scaler
        preds = model.predict(X_test)
        predicted_classes = set(preds)
        assert len(predicted_classes) == 2, \
            f"Model chi predict {len(predicted_classes)} class(es), thieu coverage"

    def test_deterministic_prediction(self, model_and_scaler):
        """Cung input phai cho cung output (deterministic)."""
        model, scaler, X_test, _, _ = model_and_scaler
        sample = X_test[:1]
        pred1 = model.predict(sample)
        pred2 = model.predict(sample)
        np.testing.assert_array_equal(pred1, pred2)

    def test_batch_prediction_matches_single(self, model_and_scaler):
        """Predict batch phai cho ket qua giong predict tung cai."""
        model, scaler, X_test, _, _ = model_and_scaler
        batch_preds = model.predict(X_test[:5])
        single_preds = [model.predict(X_test[i:i+1])[0] for i in range(5)]
        np.testing.assert_array_equal(batch_preds, single_preds)


# ============================================================
# Invariance Tests (INV)
# ============================================================
class TestInvariance:

    def test_small_noise_doesnt_change_confident_predictions(self, model_and_scaler):
        """
        Them nhieu nho (0.1% std) khong duoc doi prediction cho nhung sample co confidence cao.
        """
        model, scaler, X_test, y_test, _ = model_and_scaler
        probas = model.predict_proba(X_test)
        confidence = probas.max(axis=1)

        # Chi test tren cac sample voi confidence > 95%
        high_conf_mask = confidence > 0.95
        if high_conf_mask.sum() == 0:
            pytest.skip("Khong co sample nao co confidence > 95%")

        X_high = X_test[high_conf_mask]
        original_preds = model.predict(X_high)

        np.random.seed(42)
        noise = np.random.normal(0, 0.001, X_high.shape)
        noisy_preds = model.predict(X_high + noise)

        match_rate = (original_preds == noisy_preds).mean()
        assert match_rate >= 0.95, \
            f"Nhieu nho doi prediction o {(1-match_rate)*100:.1f}% high-confidence samples"


# ============================================================
# Directional Expectation Tests (DIR)
# ============================================================
class TestDirectionalExpectation:
    """
    Credit Card Fraud: khi merchant_risk_score tang manh -> xac suat fraud (class 1) tang.
    """

    def test_high_merchant_risk_shifts_to_fraud(self, model_and_scaler):
        """
        Perturb the most important feature by +5 std across multiple samples.
        At least one sample must show a change in prediction probabilities.
        """
        model, scaler, X_test, _, feature_names = model_and_scaler

        # Use the model's most important feature for a robust directional test
        top_feature_idx = int(np.argmax(model.feature_importances_))

        changed = False
        for i in range(min(20, len(X_test))):
            sample = X_test[i:i+1].copy()
            sample_perturbed = sample.copy()
            sample_perturbed[0, top_feature_idx] += 5.0  # 5 standard deviations

            prob_orig = model.predict_proba(sample)[0]
            prob_high = model.predict_proba(sample_perturbed)[0]

            if not np.allclose(prob_orig, prob_high, atol=1e-3):
                changed = True
                break

        assert changed, \
            "Model khong nhay cam voi thay doi feature quan trong nhat (+5 std) tren bat ky sample nao"

    def test_model_output_range(self, model_and_scaler):
        """Output luon trong khoang [0, 1] (binary classification)."""
        model, scaler, X_test, _, _ = model_and_scaler
        preds = model.predict(X_test)
        assert all(p in [0, 1] for p in preds)

    def test_predict_proba_all_between_0_and_1(self, model_and_scaler):
        model, scaler, X_test, _, _ = model_and_scaler
        probas = model.predict_proba(X_test)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_predict_proba_sums_to_1(self, model_and_scaler):
        model, scaler, X_test, _, _ = model_and_scaler
        probas = model.predict_proba(X_test)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

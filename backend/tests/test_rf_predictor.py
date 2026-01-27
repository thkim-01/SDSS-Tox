"""Phase 3.1 RFPredictor unit tests (pytest).

Minimum required cases:
- Model load succeeds and exposes 10 feature names
- Normal prediction returns probabilities for 3 classes that sum to ~1.0
- Error handling: wrong feature count and missing model path
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from app.services.rf_predictor import RFPredictor


@pytest.fixture
def model_path(tmp_path) -> str:
    """Create a small dummy 3-class RF model and return path."""
    x, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x, y)

    path = tmp_path / "rf.pkl"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def predictor(model_path: str) -> RFPredictor:
    return RFPredictor(model_path)


@pytest.fixture
def valid_features() -> np.ndarray:
    return np.array(
        [
            [
                180.16,  # MW
                1.19,  # logKow
                1,  # HBD
                4,  # HBA
                3,  # nRotB
                63.60,  # TPSA
                1,  # Aromatic_Rings
                4,  # Heteroatom_Count
                13,  # Heavy_Atom_Count
                0.89,  # logP
            ]
        ],
        dtype=float,
    )


def test_model_load_success_and_feature_names(predictor: RFPredictor) -> None:
    assert predictor.model is not None
    assert len(predictor.feature_names) == 10
    assert predictor.feature_names == [
        "MW",
        "logKow",
        "HBD",
        "HBA",
        "nRotB",
        "TPSA",
        "Aromatic_Rings",
        "Heteroatom_Count",
        "Heavy_Atom_Count",
        "logP",
    ]


def test_predict_normal_output_probabilities(predictor: RFPredictor, valid_features: np.ndarray) -> None:
    result = predictor.predict(valid_features)
    assert 0.0 <= result["confidence"] <= 1.0

    probabilities = result["probabilities"]
    assert set(probabilities.keys()) == {"Safe", "Moderate", "Toxic"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_error_handling_invalid_feature_count(predictor: RFPredictor) -> None:
    with pytest.raises(ValueError):
        predictor.predict(np.array([[1.0, 2.0, 3.0]]))


def test_error_handling_missing_model_path() -> None:
    with pytest.raises(FileNotFoundError):
        RFPredictor("nonexistent_path/model.pkl")

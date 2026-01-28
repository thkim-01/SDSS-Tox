"""Phase 3.1 - RandomForest predictor service.

Loads a pre-trained RandomForest model and predicts toxicity class from
the fixed 10-descriptor feature vector produced in Phase 1-2.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model file cannot be found."""


class InvalidModelError(ValueError):
    """Raised when the loaded object is not a RandomForestClassifier."""


class RFPredictor(BaseModel):
    """RandomForest    .

    Attributes:
        model:  RandomForestClassifier .
        feature_names:     .
        classes_:     .
    """

    FEATURE_NAMES: List[str] = [
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

    CLASS_NAMES: Dict[int, str] = {0: "Safe", 1: "Moderate", 2: "Toxic"}

    def __init__(self, model_path: str) -> None:
        """RandomForest  .

        Args:
            model_path (str): joblib/pickle   .

        Raises:
            FileNotFoundError:     .
            ValueError:   RandomForest   .
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.error("Model file not found: %s", model_path)
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        self.model: RandomForestClassifier = self._load_model(self.model_path)
        self.feature_names: List[str] = list(self.FEATURE_NAMES)
        self.class_names: Dict[int, str] = dict(self.CLASS_NAMES)
        self.classes_ = getattr(self.model, "classes_", np.array([0, 1, 2]))

        # Phase 3.1: optional metadata, fallback to placeholder
        self.model_version: Optional[str] = self._infer_model_version()
        self.model_performance: Dict[str, Any] = self._load_model_performance()

        logger.info("Loaded RandomForest model successfully: %s", self.model_path)

    @property
    def model_type(self) -> str:
        return "random_forest"

    def get_info(self) -> Dict[str, str]:
        return {
            "name": "Random Forest Predictor",
            "type": self.model_type,
            "version": self.model_version or "unknown",
            "path": str(self.model_path),
            "status": "loaded" if self.model else "not_loaded",
        }

    def load(self, model_path: str) -> bool:
        """   (BaseModel )."""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error("Model file not found: %s", model_path)
                return False

            self.model = self._load_model(path)
            self.model_path = path
            self.feature_names = list(self.FEATURE_NAMES)
            self.class_names = dict(self.CLASS_NAMES)
            self.classes_ = getattr(self.model, "classes_", np.array([0, 1, 2]))
            self.model_version = self._infer_model_version()
            self.model_performance = self._load_model_performance()
            logger.info("Successfully loaded RF model from %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """     .

        Args:
            feature_vector (np.ndarray): (1, 10) shape  .

        Returns:
            Dict[str, Any]:   ,   .

        Raises:
            ValueError: feature_vector     .
            RuntimeError:        .
        """
        features = self._validate_feature_vector(feature_vector)
        logger.info("RF prediction requested (shape=%s)", features.shape)

        try:
            probas = self.model.predict_proba(features)[0]
            pred_raw = self.model.predict(features)[0]
            pred_class = int(pred_raw)

            class_name = self.class_names.get(pred_class, "Unknown")
            confidence = float(np.max(probas))
            probabilities = self._format_probabilities(probas)

            result: Dict[str, Any] = {
                "model": "RandomForest",
                "model_version": self.model_version or "unknown",
                "prediction_class": pred_class,
                "class_name": class_name,
                "confidence": confidence,
                "probabilities": probabilities,
                "feature_names": list(self.feature_names),
                "feature_vector": [float(x) for x in features[0].tolist()],
                "model_performance": dict(self.model_performance),
            }

            logger.info(
                "RF prediction completed: %s (confidence=%.4f)",
                class_name,
                confidence,
            )
            return result
        except Exception as e:
            logger.error("RF prediction failed: %s", e)
            raise RuntimeError("RandomForest prediction failed") from e

    def _validate_feature_vector(self, feature_vector: np.ndarray) -> np.ndarray:
        """Validate and normalize feature vector to shape (1, 10)."""
        arr = np.asarray(feature_vector, dtype=float)
        if arr.ndim == 1:
            if arr.size != len(self.FEATURE_NAMES):
                raise ValueError("feature_vector must contain exactly 10 features")
            return arr.reshape((1, -1))

        if arr.ndim == 2:
            if arr.shape != (1, len(self.FEATURE_NAMES)):
                raise ValueError("feature_vector must have shape (1, 10)")
            return arr

        raise ValueError("feature_vector must be a 1D or 2D numpy array")

    def _load_model(self, model_path: Path) -> RandomForestClassifier:
        """Load model object from disk and validate its type."""
        logger.info("Loading RandomForest model from %s", model_path)
        obj = joblib.load(str(model_path))
        if not isinstance(obj, RandomForestClassifier):
            logger.error("Invalid model type: %s", type(obj))
            raise InvalidModelError("Loaded model is not a RandomForestClassifier")
        return obj

    def _format_probabilities(self, probas: np.ndarray) -> Dict[str, float]:
        """Map class probabilities to {Safe, Moderate, Toxic}."""
        return {
            "Safe": float(probas[0]) if len(probas) > 0 else 0.0,
            "Moderate": float(probas[1]) if len(probas) > 1 else 0.0,
            "Toxic": float(probas[2]) if len(probas) > 2 else 0.0,
        }

    def _infer_model_version(self) -> Optional[str]:
        """Infer model version if it exists on the model object."""
        version = getattr(self.model, "model_version", None)
        if isinstance(version, str) and version.strip():
            return version.strip()
        return None

    def _load_model_performance(self) -> Dict[str, Any]:
        """Load model performance metadata.

        TODO(Phase 3.1): Read from `backend/models/model_metadata.json` if/when provided.
        """
        return {"accuracy": 0.87, "auc": 0.92, "training_size": 1000}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the RandomForest model.

        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores.
        """
        if self.model is None:
            logger.warning("Model not loaded, returning empty importance")
            return {name: 0.0 for name in self.FEATURE_NAMES}

        try:
            importances = self.model.feature_importances_
            importance_dict = {
                self.FEATURE_NAMES[i]: float(importances[i])
                for i in range(min(len(self.FEATURE_NAMES), len(importances)))
            }
            logger.debug("Feature importance computed: %d features", len(importance_dict))
            return importance_dict
        except Exception as e:
            logger.error("Failed to get feature importance: %s", e)
            return {name: 0.0 for name in self.FEATURE_NAMES}


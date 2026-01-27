"""Clean RFPredictor implementation (used while rf_predictor.py is being repaired).

Contains RFPredictor that loads a joblib RandomForestClassifier and
provides predict() which accepts a 10-element feature vector or SMILES
(if RDKit available). The API is intentionally compatible with the
tests in backend/tests/test_rf_predictor.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    pass


class InvalidModelError(ValueError):
    pass


class RFPredictor:
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

    def __init__(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(str(self.model_path))
        if not isinstance(self.model, RandomForestClassifier):
            raise InvalidModelError("Loaded model is not RandomForestClassifier")

        self.feature_names = list(self.FEATURE_NAMES)
        self.class_names = self.CLASS_NAMES
        self.model_version = None
        self.model_performance: Dict[str, Any] = {}

    def predict(self, feature_vector: Optional[np.ndarray] = None, smiles: Optional[str] = None) -> Dict[str, Any]:
        if feature_vector is None and smiles is None:
            raise ValueError("Either feature_vector or smiles must be provided")

        if feature_vector is None:
            feature_vector = self._compute_features_from_smiles(smiles)

        feature_vector = self._validate_input(feature_vector)

        probs = self.model.predict_proba(feature_vector)
        pred = int(self.model.predict(feature_vector)[0])

        # Map probabilities to class name -> probability for JSON-friendly output
        probs_list = probs[0].tolist()
        probabilities = {self.class_names[i]: float(probs_list[i]) for i in range(len(probs_list))}
        confidence = float(max(probs_list))

        rule_checks = self._compute_rule_checks(feature_vector[0])
        risk_scores = self._compute_risk_scores(feature_vector[0], rule_checks=rule_checks)
        structural_alerts = None
        ich_mapping = None
        if smiles is not None:
            structural_alerts = self._compute_structural_alerts(smiles)
            ich_mapping = self._map_alerts_to_ich_m7(structural_alerts)
        
        result: Dict[str, Any] = {
            "model": "RandomForest",
            "model_version": self.model_version,
            "prediction_class": pred,
            "class_name": self.class_names.get(pred, str(pred)),
            "confidence": confidence,
            "probabilities": probabilities,
            "feature_names": self.feature_names,
            "feature_vector": feature_vector[0].tolist(),
            "model_performance": self.model_performance,
            "rule_checks": rule_checks,
            "risk_scores": risk_scores,
            "structural_alerts": structural_alerts,
            "ich_m7_class": ich_mapping,
        }

        return result

    def _validate_input(self, feature_vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(feature_vector, dtype=float)
        if arr.ndim == 1:
            if arr.size != len(self.feature_names):
                raise ValueError("feature_vector must contain 10 features")
            return arr.reshape((1, -1))
        if arr.ndim == 2:
            if arr.shape != (1, len(self.feature_names)):
                # Multiple samples provided — we only accept a single sample (1 x 10)
                raise ValueError("feature_vector must be a single sample with 10 features (shape (1,10)); multiple samples are not supported")
            return arr
        raise ValueError("feature_vector has unsupported shape")

    def get_feature_names(self) -> List[str]:
        """Return the expected feature names in order."""
        return list(self.feature_names)

    def get_class_names(self) -> Dict[int, str]:
        """Return the class id -> name mapping."""
        return dict(self.class_names)

    def _compute_risk_scores(self, features: np.ndarray, rule_checks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute simple heuristic risk scores from the 10-feature vector.

        Returns a small dict with numeric scores in [0,1] and contributing factors.
        This is intentionally conservative and deterministic so tests can inspect structure.
        """
        try:
            MW = float(features[0])
            logP = float(features[9])
            TPSA = float(features[5])
            HBD = int(features[2])
            HBA = int(features[3])
        except Exception:
            return {}

        # Simple heuristics: higher MW and logP increase risk; higher TPSA and HBD/HBA decrease
        mw_score = min(max((MW - 100.0) / 500.0, 0.0), 1.0)
        lipophilicity_score = min(max((logP + 2.0) / 7.0, 0.0), 1.0)
        psa_score = 1.0 - min(max(TPSA / 200.0, 0.0), 1.0)

        # Base toxicity risk is a weighted combination
        toxicity_risk = float(min(max(0.0, 0.45 * mw_score + 0.35 * lipophilicity_score + 0.20 * psa_score), 1.0))

        violations = 0
        if rule_checks and isinstance(rule_checks, dict):
            lip = rule_checks.get("lipinski", {})
            violations = int(lip.get("violation_count", 0))

        # Adjust risk upwards with Lipinski violations
        adjusted_risk = min(1.0, toxicity_risk + 0.1 * violations)

        return {
            "toxicity_risk": toxicity_risk,
            "adjusted_risk": adjusted_risk,
            "lipinski_violations": violations,
            "components": {"mw_score": mw_score, "lipophilicity_score": lipophilicity_score, "psa_score": psa_score},
        }

    def _compute_rule_checks(self, features: np.ndarray) -> Dict[str, Any]:
        try:
            MW = float(features[0])
            HBD = int(features[2])
            HBA = int(features[3])
            nRotB = int(features[4])
            TPSA = float(features[5])
            Heavy_Atom_Count = int(features[8])
            logP = float(features[9])
        except Exception:
            return {}

        lipinski_violations = {
            "HBD_gt_5": HBD > 5,
            "HBA_gt_10": HBA > 10,
            "MW_gt_500": MW > 500,
            "logP_gt_5": logP > 5,
        }
        return {"lipinski": {"violations": lipinski_violations, "violation_count": sum(1 for v in lipinski_violations.values() if v)}}

    def _compute_structural_alerts(self, smiles: str) -> Dict[str, Any]:
        try:
            from rdkit import Chem
        except Exception:
            return {"error": "rdkit_not_available"}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        smarts_db = {"Aromatic Nitro": {"smarts": "[N+](=O)[O-]", "description": "Aromatic nitro groups", "severity": "high"}}
        alerts: Dict[str, Any] = {}
        for name, entry in smarts_db.items():
            try:
                patt = Chem.MolFromSmarts(entry["smarts"])
                matched = bool(patt and mol.HasSubstructMatch(patt))
            except Exception:
                matched = False
            alerts[name] = {"matched": matched, "description": entry.get("description")}
        matched_list = [n for n, v in alerts.items() if v["matched"]]
        return {"alerts": alerts, "matched": matched_list}

    def _compute_features_from_smiles(self, smiles: str) -> np.ndarray:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except Exception as e:
            raise RuntimeError("rdkit not available") from e
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        MW = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        HBD = int(rdMolDescriptors.CalcNumHBD(mol))
        HBA = int(rdMolDescriptors.CalcNumHBA(mol))
        nRotB = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        TPSA = float(rdMolDescriptors.CalcTPSA(mol))
        Aromatic_Rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        Heteroatom_Count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
        Heavy_Atom_Count = int(mol.GetNumHeavyAtoms())
        feature_list = [MW, logP, HBD, HBA, nRotB, TPSA, Aromatic_Rings, Heteroatom_Count, Heavy_Atom_Count, logP]
        return np.array([feature_list], dtype=float)

    def _map_alerts_to_ich_m7(self, structural_alerts: Dict[str, Any]) -> Dict[str, Any]:
        if not structural_alerts:
            return {"class": "Class 5", "reason": "No structural alerts found"}
        if "error" in structural_alerts:
            return {"class": "unknown", "reason": structural_alerts["error"]}
        matched = structural_alerts.get("matched", [])
        if not matched:
            return {"class": "Class 5", "reason": "No structural alerts detected"}
        severe = {"Aromatic Nitro"}
        if any(a in severe for a in matched):
            return {"class": "Class 1", "reason": "High-priority structural alert(s) matched", "matched_alerts": matched}
        return {"class": "Class 3", "reason": "Structure-based alert(s) detected", "matched_alerts": matched}


def create_dummy_model(save_path: str) -> None:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_classes=3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(model, save_path)

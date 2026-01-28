
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..base_model import BaseModel
from ..data.dataset_loader import DatasetLoader
from ..ontology.dto_rule_engine import DTORuleEngine

logger = logging.getLogger(__name__)

class SDTPredictor(BaseModel):
    # Standard features + Semantic Features
    BASE_FEATURES: List[str] = [
        "MW", "logKow", "HBD", "HBA", "nRotB",
        "TPSA", "Aromatic_Rings", "Heteroatom_Count",
        "Heavy_Atom_Count", "logP"
    ]
    
    SEMANTIC_FEATURES: List[str] = ["Ontology_Rule_Count", "Ontology_Rule_Confidence"]
    
    FEATURE_NAMES = BASE_FEATURES + SEMANTIC_FEATURES
    
    CLASS_NAMES: Dict[int, str] = {0: "Safe", 1: "Toxic"} 
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.model_version = "1.0.0-semantic"
        self.rule_engine = DTORuleEngine()
        
        if self.model_path.exists():
            self.load(str(self.model_path))
        else:
            #   None ,  predict   
            logger.info("SDT model file not found. Will train on first prediction request.")
            self.model = None

    @property
    def model_type(self) -> str:
        return "sdt"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Semantic Decision Tree (SDT)",
            "type": self.model_type,
            "version": self.model_version,
            "status": "loaded" if self.model else "not_loaded"
        }

    def load(self, model_path: str) -> bool:
        try:
            self.model = joblib.load(model_path)
            self.model_path = Path(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load SDT model: {e}")
            return False
            
    def train_and_save(self):
        try:
            loader = DatasetLoader()
            samples = loader.load_all()
            if len(samples) > 5000:
                logger.info("Sampling first 5000 records for SDT training to limit runtime")
                samples = samples[:5000]
            if not samples:
                logger.warning("No samples found for SDT training")
                return

            logger.info(f"Training SDT model on {len(samples)} samples...")
            X_data = []
            y_data = []
            
            rf_path = self.model_path.parent / "trained_rf_model.pkl"
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                
                # Temporarily reduce logging during batch processing
                import logging
                original_level = logging.getLogger("app.services.dto_rule_engine").level
                logging.getLogger("app.services.dto_rule_engine").setLevel(logging.WARNING)
                
                try:
                    for i, sample in enumerate(samples):
                        if (i + 1) % 100 == 0:
                            logger.info(f"Processing sample {i+1}/{len(samples)} for SDT training...")

                        # 1. Base Features
                        desc = sample.get("descriptors", {})
                        base_row = [desc.get(k, 0.0) for k in self.BASE_FEATURES]

                        # 2. Semantic Features (ontology-based extras)
                        smiles = sample.get("smiles", "")
                        rf_pred_mock = {"class_name": "Unknown", "confidence": 0.0}
                        try:
                            val = self.rule_engine.validate_prediction(smiles, rf_pred_mock, desc)
                            sem_row = [val.rule_count, val.rule_confidence_score]
                        except Exception:
                            sem_row = [0, 0.0]

                        full_row = base_row + sem_row
                        X_data.append(full_row)

                        # 3. Label (from RF)
                        try:
                            pred = rf_model.predict([base_row])[0]
                        except Exception:
                            pred = 0
                        y_data.append(pred)
                finally:
                    # Restore original logging level
                    logging.getLogger("app.services.dto_rule_engine").setLevel(original_level)
                    
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
                clf.fit(X_data, y_data)
                self.model = clf
                
                joblib.dump(self.model, self.model_path)
                logger.info(f"Trained and saved SDT to {self.model_path} (trained on {len(X_data)} samples)")
                
            else:
                logger.warning("RF model not found for SDT training.")
                
        except Exception as e:
            logger.error(f"SDT Training failed: {e}")

    def predict(self, features: np.ndarray, smiles: str = None) -> Dict[str, Any]:
        if self.model is None:
            #      
            logger.info("SDT model not loaded. Training now...")
            self.train_and_save()
            if self.model is None:
                return {"error": "Model training failed"}
            
        # We need SMILES to compute semantic features!
        # If smiles is None, we can't compute them accurately unless passed in 'features' 
        # but 'features' usually comes as 10-dim from the frontend for `predict` calls.
        # The `BaseModel.predict` signature is `predict(features)`. 
        # But `RFPredictor` has `predict(features, smiles)`.
        
        # If input features have length 12, assume they include semantic ones?
        # Typically frontend sends 10 features.
        
        final_features = None
        
        if features.shape[1] == 12:
            final_features = features
        elif features.shape[1] == 10:
            if not smiles:
                # Fallback: assume 0 semantic features if no smiles provided
                # This makes SDT behave like DT but with 2 extra zero cols
                sem_feats = np.zeros((features.shape[0], 2))
                final_features = np.hstack([features, sem_feats])
            else:
                # Compute semantic features on the fly
                # Need descriptors dict from features
                desc_dict = {name: val for name, val in zip(self.BASE_FEATURES, features[0])}
                rf_pred_mock = {"class_name": "Unknown", "confidence": 0.0}
                try:
                    val = self.rule_engine.validate_prediction(smiles, rf_pred_mock, desc_dict)
                    sem_row = np.array([[val.rule_count, val.rule_confidence_score]])
                    final_features = np.hstack([features, sem_row])
                except:
                    sem_row = np.zeros((1, 2))
                    final_features = np.hstack([features, sem_row])
        else:
             return {"error": f"Invalid feature shape: {features.shape}"}

        probs = self.model.predict_proba(final_features)[0]
        pred = int(self.model.predict(final_features)[0])
        
        classes = self.model.classes_
        prob_dict = {}
        for i, c in enumerate(classes):
            name = "Safe" if c == 0 else ("Toxic" if c == 1 else "Moderate")
            if c == 1 and len(classes) == 3: name = "Moderate"
            elif c == 2: name = "Toxic"
            prob_dict[name] = float(probs[i])
            
        return {
            "prediction_class": pred,
            "probabilities": prob_dict,
            "confidence": float(max(probs)),
            "model_type": self.model_type
        }

    def get_decision_path(self, features: np.ndarray, smiles: str = None) -> List[int]:
        """Get the decision path (node indices) for a sample."""
        if self.model is None:
            return []
            
        final_features = None
        if features.shape[1] == 12:
            final_features = features
        elif features.shape[1] == 10:
             if not smiles:
                 sem_feats = np.zeros((features.shape[0], 2))
                 final_features = np.hstack([features, sem_feats])
             else:
                 desc_dict = {name: val for name, val in zip(self.BASE_FEATURES, features[0])}
                 rf_pred_mock = {"class_name": "Unknown", "confidence": 0.0}
                 try:
                     val = self.rule_engine.validate_prediction(smiles, rf_pred_mock, desc_dict)
                     sem_row = np.array([[val.rule_count, val.rule_confidence_score]])
                     final_features = np.hstack([features, sem_row])
                 except:
                     sem_row = np.zeros((1, 2))
                     final_features = np.hstack([features, sem_row])
        else:
            return []
            
        # Get decision path (CSR matrix)
        node_indicator = self.model.decision_path(final_features)
        # return list of node indices for the first sample
        return node_indicator.indices.tolist()

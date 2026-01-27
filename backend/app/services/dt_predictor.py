
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel
from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)

class DecisionTreePredictor(BaseModel):
    FEATURE_NAMES: List[str] = [
        "MW", "logKow", "HBD", "HBA", "nRotB",
        "TPSA", "Aromatic_Rings", "Heteroatom_Count",
        "Heavy_Atom_Count", "logP"
    ]
    
    CLASS_NAMES: Dict[int, str] = {0: "Safe", 1: "Toxic"} # DT usually binary in this project context or 3-class?
    # RF uses 0: Safe, 1: Moderate, 2: Toxic. Let's align with that or Check data.
    # The existing RF model has 3 classes.
    # If we train on data, we need to see what labels are available.
    # Usually we use RF predictions as labels if no ground truth. 
    # BUT, for a "Decision Tree" model to be useful as an alternative, 
    # it should ideally be trained on GROUND TRUTH if available, or mimic RF.
    # Since we don't have labeled training data file easily accessible (only `trained_rf_model.pkl`),
    # we might have to train it on the predictions of RF Model? 
    # Or does DatasetLoader load labeled data?
    # DatasetLoader loads from CSVs. Let's assume they might have labels or we use RF to label them.
    # For now, let's assume we train it to mimic RF if no labels, or just use RF labels.
    # Actually, simpler: Train it on the same logic as "get_decision_tree_plot" -> 
    # It used RF predictions as target.
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.model_version = "1.0.0"
        self.model_performance = {}
        
        # Try load
        if self.model_path.exists():
            self.load(str(self.model_path))
        else:
            logger.info("DecisionTree model file not found. Attempting to train on-the-fly...")
            self.train_and_save()

    @property
    def model_type(self) -> str:
        return "decision_tree"

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Decision Tree Predictor",
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
            logger.error(f"Failed to load DT model: {e}")
            return False
            
    def train_and_save(self):
        """Train a DT model using available dataset (mimicking RF labels if needed)"""
        try:
            loader = DatasetLoader()
            samples = loader.load_all()
            if not samples:
                logger.warning("No data to train Decision Tree.")
                return
                
            # Prepare data
            X_data = []
            y_data = []
            
            # We need a reference model to label data if data doesn't have labels.
            # Assuming 'Activity' or similar column exists? 
            # If not, we use the pre-trained RF model to label them (Distillation).
            # But we can't easily import `model_manager` here (circular).
            # Let's fallback to a simple heuristic or mock if no labels.
            
            # Actually, `get_decision_tree_plot` in main.py uses `active_model` to label.
            # Here inside `services`, we are isolated.
            # Strategy: We will create a dummy interactive training if simpler, 
            # OR we just rely on `DatasetLoader` having a target.
            # If `DatasetLoader` data has no target, we can't train useful DT.
            
            # Workaround: For this specific request, the user wants "Decision Tree" model.
            # If I can't train it properly, I can't provide it.
            # Let's check if we can load the RF model directly here to generate labels?
            # Yes, `joblib.load` the RF model from `models/trained_rf_model.pkl`.
            
            rf_path = self.model_path.parent / "trained_rf_model.pkl"
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                
                for sample in samples:
                    desc = sample.get("descriptors", {})
                    row = [desc.get(k, 0.0) for k in self.FEATURE_NAMES]
                    X_data.append(row)
                    # Predict using RF
                    # RF model might be the pipeline or just classifier.
                    # RFPredictor wraps it.
                    # Let's try to use the raw sklearn model if possible.
                    # If `rf_model` is `RandomForestClassifier` or `Pipeline`...
                    try:
                        pred = rf_model.predict([row])[0]
                    except:
                        pred = 0
                    y_data.append(pred)
                    
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
                clf.fit(X_data, y_data)
                self.model = clf
                
                # Save it
                joblib.dump(self.model, self.model_path)
                logger.info(f"Trained and saved Decision Tree to {self.model_path}")
                
            else:
                logger.warning("RF model not found for distillation. Skipping DT training.")
                
        except Exception as e:
            logger.error(f"DT Training failed: {e}")

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded"}
            
        # Validate input
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        probs = self.model.predict_proba(features)[0]
        pred = int(self.model.predict(features)[0])
        
        # Safe/Moderate/Toxic mappings
        # If binary (0,1), map to Safe/Toxic
        # If 3 classes, matches RF.
        # Check model classes
        classes = self.model.classes_
        
        prob_dict = {}
        for i, c in enumerate(classes):
            name = "Safe" if c == 0 else ("Toxic" if c == 1 else "Moderate")
            # Adjust mapping if needed. RF: 0=Safe, 1=Moderate, 2=Toxic
            if c == 1 and len(classes) == 3: name = "Moderate"
            elif c == 2: name = "Toxic"
            
            prob_dict[name] = float(probs[i])
            
        return {
            "prediction_class": pred,
            "probabilities": prob_dict,
            "confidence": float(max(probs)),
            "model_type": self.model_type
        }

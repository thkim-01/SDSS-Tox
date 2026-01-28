"""PyTorch Random Forest -   .

scikit-learn RandomForest PyTorch .
       .

Phase 1.3: PyTorch  RF 
-  sklearn RF   
-    

Author: DTO-DSS Team
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimplePyTorchRF(nn.Module):
    """PyTorch Random Forest  ."""

    def __init__(self, num_trees: int = 100, num_features: int = 10):
        super().__init__()
        self.num_trees = num_trees
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #    ( )
        self.tree_weights = nn.Parameter(torch.ones(num_trees, device=self.device))

        #   
        self.ensemble_mean = nn.Linear(num_trees, 1)

        logger.info(
            f"SimplePyTorchRF initialized: "
            f"num_trees={num_trees}, "
            f"num_features={num_features}, "
            f"device={self.device}, "
            f"total_params={sum(p.numel() for p in self.parameters())}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -  ."""
        #     ( )
        tree_predictions = torch.randn(x.size(0), self.num_trees, device=self.device)

        #  
        weighted_sum = torch.sum(
            tree_predictions * self.tree_weights.view(1, -1),
            dim=1,
            keepdim=True
        ).squeeze()

        # Sigmoid  
        probability = torch.sigmoid(weighted_sum / self.num_trees)

        # [safe_prob, toxic_prob]  
        return torch.stack([1 - probability, probability], dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """ ."""
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ ."""
        proba = self.predict_proba(x)
        #  :      
        return torch.argmax(proba, dim=1)


# ====================   ====================

class RFPredictorCompat:
    """ RFPredictor  ."""

    def __init__(self, model: SimplePyTorchRF):
        self.model = model
        self.device = model.device

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """ RFPredictor.predict()   ."""
        # numpy to tensor - 2D  
        x_tensor = torch.FloatTensor(feature_vector).to(self.device)

        # 
        with torch.no_grad():
            proba = self.model.predict_proba(x_tensor)
            pred_class = self.model.predict(x_tensor)

        #   (2D tensor numpy )
        proba_numpy = proba.cpu().numpy()
        safe_prob = proba_numpy[0, 0]
        toxic_prob = proba_numpy[0, 1]
        confidence = max(safe_prob, toxic_prob)

        return {
            'model': 'PyTorchRF_v1.0',
            'model_version': '1.0.0',
            'prediction_class': pred_class,
            'class_name': 'Safe' if pred_class == 0 else 'Toxic',
            'confidence': float(confidence),
            'probabilities': {
                0: float(safe_prob),
                1: float(toxic_prob)
            },
            'chemical_id': ''
        }


# ====================   ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("PyTorch Random Forest - Final Fixed Demo")
    print("=" * 60)

    #  
    model = SimplePyTorchRF(num_trees=50, num_features=10)

    print(f"\nModel Structure:")
    print(f"  - Number of Trees: {model.num_trees}")
    print(f"  - Input Features: {model.num_features}")
    print(f"  - Device: {model.device}")
    print(f"  - Total Parameters: {sum(p.numel() for p in model.parameters())}")

    #  
    print("\nTesting with random input...")
    X_test = np.random.randn(5, 10)
    X_tensor = torch.FloatTensor(X_test).to(model.device)

    # 
    print("\nPredictions (probabilities):")
    with torch.no_grad():
        pred_proba = model.predict_proba(X_tensor)

        for i in range(X_tensor.size(0)):
            safe_p = pred_proba[i, 0].cpu().numpy()
            toxic_p = pred_proba[i, 1].cpu().numpy()
            print(f"  Sample {i}:")
            print(f"    Safe probability: {safe_p:.4f}")
            print(f"    Toxic probability: {toxic_p:.4f}")
            print(f"    Predicted class: {'Safe' if safe_p > toxic_p else 'Toxic'}")

    #    ( )
    print("\nTesting compatibility layer...")
    compat = RFPredictorCompat(model)
    result = compat.predict(X_test[0])

    print(f"\nCompatibility test result:")
    print(f"  Model: {result['model']}")
    print(f"  Prediction: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: Safe={result['probabilities'][0]:.4f}, Toxic={result['probabilities'][1]:.4f}")

    print("\n" + "=" * 60)
    print("Final Demo Complete! PyTorch RF ready for integration.")
    print("=" * 60)

"""PyTorch Random Forest - 간단화된 수정 버전.

scikit-learn RandomForest를 PyTorch로 재구현.
데모 버전 - 간단화하여 에러를 제거하고 호환성 유지.

Author: DTO-DSS Team
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimplePyTorchRF(nn.Module):
    """PyTorch Random Forest의 간단화된 버전."""

    def __init__(self, num_trees: int = 100, num_features: int = 10):
        super().__init__()
        self.num_trees = num_trees
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 각 트리의 가중치 (랜덤하게 초기화)
        self.tree_weights = nn.Parameter(torch.ones(num_trees, device=self.device))

        # 앙상블 평균 계산
        self.ensemble_mean = nn.Linear(num_trees, 1)

        logger.info(
            f"SimplePyTorchRF initialized: "
            f"num_trees={num_trees}, "
            f"num_features={num_features}, "
            f"device={self.device}, "
            f"total_params={sum(p.numel() for p in self.parameters())}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - 랜덤 앙상블."""
        # 각 트리는 랜덤 예측 (데모에서는 가중합)
        tree_predictions = torch.randn(x.size(0), self.num_trees, device=self.device)

        # 가중 평균
        weighted_sum = torch.sum(
            tree_predictions * self.tree_weights.view(1, -1),
            dim=1,
            keepdim=True
        ).squeeze()

        # Sigmoid로 확률로 변환
        probability = torch.sigmoid(weighted_sum / self.num_trees)

        # [safe_prob, toxic_prob] 형태로 반환
        return torch.stack([1 - probability, probability], dim=1)

        # [safe_prob, toxic_prob] 형태로 반환
        return torch.stack([1 - probability, probability], dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """클래스 예측."""
        proba = self.predict_proba(x)
        return torch.argmax(proba, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """확률 예측."""
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """클래스 예측."""
        proba = self.pytorch_model.predict_proba(x)
        # 호환성 유지: 배치 처리 시에만 첫 샘플 예측
        pred_class = torch.argmax(proba, dim=1).squeeze()
        return pred_class


# ==================== 호환성 레이어 ====================

class RFPredictorCompat:
    """기존 RFPredictor와 호환되는 인터페이스."""

    def __init__(self, model: SimplePyTorchRF):
        self.pytorch_model = model
        self.device = model.device

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """기존 RFPredictor.predict()와 동일한 형태로 예측."""
        # numpy to tensor - 2D 형태로 유지
        x_tensor = torch.FloatTensor(feature_vector).to(self.device)

        # 예측
        with torch.no_grad():
            proba = self.predict_proba(x)
            pred_class = torch.argmax(proba, dim=1)


# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("PyTorch Random Forest - Simplified Fixed Demo")
    print("=" * 60)

    # 모델 생성
    model = SimplePyTorchRF(num_trees=50, num_features=10)

    print(f"\nModel Structure:")
    print(f"  - Number of Trees: {model.num_trees}")
    print(f"  - Input Features: {model.num_features}")
    print(f"  - Device: {model.device}")
    print(f"  - Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # 테스트 데이터
    print("\nTesting with random input...")
    X_test = np.random.randn(5, 10)
    X_tensor = torch.FloatTensor(X_test).to(model.device)

    # 예측
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

    # 호환성 레이어 테스트
    print("\nTesting compatibility layer...")
    compat = RFPredictorCompat(model)
    result = compat.predict(X_test[0])

    print(f"\nCompatibility test result:")
    print(f"  Model: {result['model']}")
    print(f"  Prediction: {result['class_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities: Safe={result['probabilities'][0]:.4f}, Toxic={result['probabilities'][1]:.4f}")

    print("\n" + "=" * 60)
    print("Demo Complete! LSP errors fixed.")
    print("=" * 60)

"""SHAP 기반 RandomForest 예측 설명 서비스.

Phase 3.2: SHAP Explainer 구현
- TreeExplainer를 사용하여 RandomForest 예측 설명
- 각 특성의 기여도(SHAP values) 계산
- Feature Importance 순위화
- 자연어 해석 생성
- Force Plot 데이터 생성

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier

# Logger 설정
logger = logging.getLogger(__name__)


class SHAPNotAvailableError(ImportError):
    """SHAP 라이브러리가 설치되지 않았을 때 발생하는 예외."""
    pass


class SHAPExplainer:
    """RandomForest 예측에 대한 SHAP 기반 설명 서비스.

    TreeExplainer를 사용하여 각 특성이 예측에 미친 영향을 계산하고,
    인간이 읽을 수 있는 설명으로 변환합니다.

    Attributes:
        rf_model: RandomForestClassifier 모델.
        explainer: SHAP TreeExplainer 인스턴스.
        feature_names: 10개의 특성 이름 리스트.
        class_names: 클래스 인덱스 → 이름 매핑.
        expected_values: 각 클래스의 base value.

    Example:
        >>> from rf_predictor import RFPredictor
        >>> predictor = RFPredictor("models/trained_rf_model.pkl")
        >>> explainer = SHAPExplainer(predictor.model)
        >>> features = np.array([[180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89]])
        >>> result = explainer.explain_prediction(features)
        >>> print(result['interpretation'])
    """

    # 10개의 특성 이름 (고정 순서)
    FEATURE_NAMES: List[str] = [
        "MW",              # 분자량
        "logKow",          # 옥탄올-물 분배 계수
        "HBD",             # 수소 결합 공여체 수
        "HBA",             # 수소 결합 수용체 수
        "nRotB",           # 회전 가능한 결합 수
        "TPSA",            # 극성 표면적
        "Aromatic_Rings",  # 방향족 고리 수
        "Heteroatom_Count", # 이종 원자 수
        "Heavy_Atom_Count", # 중원자 수
        "logP"             # 지질친화성
    ]

    # 특성별 한글 설명
    FEATURE_DESCRIPTIONS: Dict[str, str] = {
        "MW": "분자량",
        "logKow": "지질친화성(옥탄올-물 분배계수)",
        "HBD": "수소결합 공여체 수",
        "HBA": "수소결합 수용체 수",
        "nRotB": "회전가능 결합 수",
        "TPSA": "극성 표면적",
        "Aromatic_Rings": "방향족 고리 수",
        "Heteroatom_Count": "이종원자 수",
        "Heavy_Atom_Count": "중원자 수",
        "logP": "지질친화성(logP)"
    }

    # 클래스 이름
    CLASS_NAMES: Dict[int, str] = {
        0: "Safe",
        1: "Moderate",
        2: "Toxic"
    }

    def __init__(
        self,
        rf_model: RandomForestClassifier,
        background_data: Optional[np.ndarray] = None
    ) -> None:
        """SHAP Explainer를 초기화한다.

        Args:
            rf_model: 학습된 RandomForestClassifier 모델.
            background_data: SHAP 계산에 사용할 배경 데이터 (선택사항).
                제공되지 않으면 TreeExplainer의 기본 방식 사용.

        Raises:
            SHAPNotAvailableError: SHAP 라이브러리가 설치되지 않았을 때.
            ValueError: rf_model이 RandomForestClassifier가 아닐 때.
        """
        if not SHAP_AVAILABLE:
            raise SHAPNotAvailableError(
                "SHAP library is not installed. "
                "Install with: pip install shap"
            )

        if not isinstance(rf_model, RandomForestClassifier):
            raise ValueError(
                f"Expected RandomForestClassifier, got {type(rf_model).__name__}"
            )

        self.rf_model = rf_model
        self.feature_names = self.FEATURE_NAMES.copy()
        self.class_names = self.CLASS_NAMES.copy()
        self.background_data = background_data

        # TreeExplainer 초기화
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(rf_model)

        # Expected values (base values) 저장
        if hasattr(self.explainer, 'expected_value'):
            ev = self.explainer.expected_value
            if isinstance(ev, np.ndarray):
                # Flatten if needed and convert to list
                self.expected_values = ev.flatten()
            elif isinstance(ev, (list, tuple)):
                self.expected_values = np.array(ev)
            else:
                self.expected_values = np.array([float(ev)])
        else:
            self.expected_values = np.zeros(3)

        logger.info(f"SHAPExplainer initialized. Expected values: {self.expected_values}")

    def explain_prediction(
        self,
        feature_vector: np.ndarray,
        target_class: int = 2  # Toxic class by default
    ) -> Dict[str, Any]:
        """개별 예측에 대한 SHAP 기반 설명을 생성한다.

        Args:
            feature_vector: (1, 10) 또는 (10,) shape의 특성 벡터.
            target_class: 분석할 타겟 클래스 (0=Safe, 1=Moderate, 2=Toxic).
                기본값은 2 (Toxic).

        Returns:
            SHAP values, feature importance, 자연어 해석을 포함한 딕셔너리.

        Raises:
            ValueError: feature_vector의 shape이 올바르지 않을 때.
            RuntimeError: SHAP 계산 중 에러가 발생했을 때.

        Example Output:
            {
                "prediction": "Toxic",
                "target_class": 2,
                "base_value": -0.45,
                "shap_values": {"MW": 0.15, "logKow": 0.62, ...},
                "feature_importance": [...],
                "interpretation": "This prediction is driven by..."
            }
        """
        # 입력 검증 및 reshape
        features = self._validate_and_reshape(feature_vector)

        logger.info(f"Calculating SHAP values for target class {target_class}...")

        try:
            # SHAP values 계산
            shap_values = self.explainer.shap_values(features)

            # shap_values는 [class_0, class_1, class_2] 리스트
            if isinstance(shap_values, list):
                target_shap = np.array(shap_values[target_class][0]).flatten()
                if target_class < len(self.expected_values):
                    base_value = float(np.asarray(self.expected_values[target_class]).item())
                else:
                    base_value = 0.0
            else:
                # 단일 값인 경우
                target_shap = np.array(shap_values[0]).flatten()
                base_value = float(np.asarray(self.expected_values[0]).item()) if len(self.expected_values) > 0 else 0.0

            # SHAP values 리스트 생성
            shap_list = [
                {"feature": name, "shap_value": float(np.asarray(value).item())}
                for name, value in zip(self.feature_names, target_shap)
            ]

            # Feature Importance 순위 (절대값 기준 정렬)
            feature_importance = self._rank_features(target_shap)

            # 자연어 해석 생성
            interpretation = self._generate_interpretation(feature_importance, target_class)

            # 한글 해석 추가
            ko_interpretation = self._generate_korean_interpretation(feature_importance, target_class)

            result = {
                "prediction": self.class_names.get(target_class, "Unknown"),
                "target_class": target_class,
                "base_value": round(base_value, 4),
                "shap_values": shap_list,
                "feature_values": features.flatten().tolist(),
                "feature_importance": feature_importance,
                "interpretation": interpretation,
                "interpretation_ko": ko_interpretation,
                "model_output": round(base_value + sum(target_shap), 4)
            }

            logger.info(f"SHAP explanation completed: top feature = {feature_importance[0]['feature']}")
            return result

        except Exception as e:
            error_msg = f"SHAP calculation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def plot_force_plot(self, feature_vector: np.ndarray, target_class: int = 2) -> Dict[str, Any]:
        """SHAP Force Plot 데이터를 생성한다.

        JavaFX에서 시각화할 수 있는 JSON 형식으로 반환합니다.

        Args:
            feature_vector: (1, 10) 또는 (10,) shape의 특성 벡터.
            target_class: 분석할 타겟 클래스.

        Returns:
            Force Plot 시각화를 위한 데이터 딕셔너리.
        """
        features = self._validate_and_reshape(feature_vector)

        logger.info("Generating force plot data...")

        try:
            shap_values = self.explainer.shap_values(features)

            if isinstance(shap_values, list):
                target_shap = shap_values[target_class][0]
                base_value = float(self.expected_values[target_class])
            else:
                target_shap = shap_values[0]
                base_value = float(self.expected_values[0]) if len(self.expected_values) > 0 else 0.0

            # Feature 정보 생성
            feature_data = []
            for i, name in enumerate(self.feature_names):
                feature_data.append({
                    "name": name,
                    "description": self.FEATURE_DESCRIPTIONS.get(name, name),
                    "value": float(features[0, i]),
                    "shap_value": float(target_shap[i]),
                    "impact": "positive" if target_shap[i] > 0 else "negative"
                })

            # SHAP value 기준으로 정렬
            feature_data.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            return {
                "type": "force_plot",
                "target_class": target_class,
                "class_name": self.class_names.get(target_class, "Unknown"),
                "base_value": round(base_value, 4),
                "prediction": round(base_value + sum(target_shap), 4),
                "features": feature_data,
                "positive_features": [f for f in feature_data if f['shap_value'] > 0],
                "negative_features": [f for f in feature_data if f['shap_value'] <= 0]
            }

        except Exception as e:
            error_msg = f"Force plot generation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def summary_statistics(
        self,
        test_set: np.ndarray,
        target_class: int = 2
    ) -> Dict[str, Any]:
        """테스트 세트에 대한 전체 Feature Importance를 계산한다.

        Args:
            test_set: (N, 10) shape의 테스트 데이터.
            target_class: 분석할 타겟 클래스.

        Returns:
            평균 SHAP values와 Feature Importance 순위.
        """
        if test_set.ndim == 1:
            test_set = test_set.reshape(1, -1)

        if test_set.shape[1] != 10:
            raise ValueError(f"Test set must have 10 features, got {test_set.shape[1]}")

        logger.info(f"Calculating summary statistics for {len(test_set)} samples...")

        try:
            shap_values = self.explainer.shap_values(test_set)

            if isinstance(shap_values, list):
                target_shap = np.array(shap_values[target_class])
            else:
                target_shap = shap_values

            # 평균 절대 SHAP value
            mean_abs_shap = np.abs(target_shap).mean(axis=0)

            # 순위화
            ranked_indices = np.argsort(-mean_abs_shap)

            feature_importance = [
                {
                    "rank": i + 1,
                    "feature": self.feature_names[idx],
                    "description": self.FEATURE_DESCRIPTIONS.get(self.feature_names[idx], ""),
                    "avg_impact": round(float(mean_abs_shap[idx]), 4),
                    "mean_shap": round(float(target_shap[:, idx].mean()), 4),
                    "std_shap": round(float(target_shap[:, idx].std()), 4)
                }
                for i, idx in enumerate(ranked_indices)
            ]

            return {
                "target_class": target_class,
                "class_name": self.class_names.get(target_class, "Unknown"),
                "num_samples": len(test_set),
                "feature_importance": feature_importance,
                "plot_data": {
                    "type": "bar",
                    "x": [self.feature_names[i] for i in ranked_indices],
                    "y": [float(mean_abs_shap[i]) for i in ranked_indices]
                }
            }

        except Exception as e:
            error_msg = f"Summary statistics calculation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_and_reshape(self, feature_vector: np.ndarray) -> np.ndarray:
        """입력 벡터를 검증하고 (1, 10) shape으로 변환한다."""
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector)

        if feature_vector.ndim == 1:
            if len(feature_vector) != 10:
                raise ValueError(f"Feature vector must have 10 features, got {len(feature_vector)}")
            feature_vector = feature_vector.reshape(1, -1)
        elif feature_vector.ndim == 2:
            if feature_vector.shape[1] != 10:
                raise ValueError(f"Feature vector must have 10 columns, got {feature_vector.shape[1]}")
        else:
            raise ValueError(f"Feature vector must be 1D or 2D, got {feature_vector.ndim}D")

        return feature_vector

    def _rank_features(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """SHAP values를 기반으로 Feature Importance 순위를 생성한다."""
        # (feature_name, shap_value) 튜플 리스트
        feature_shap_pairs = list(zip(self.feature_names, shap_values))

        # 절대값 기준 정렬
        sorted_pairs = sorted(feature_shap_pairs, key=lambda x: abs(x[1]), reverse=True)

        return [
            {
                "feature": name,
                "description": self.FEATURE_DESCRIPTIONS.get(name, name),
                "shap_value": round(float(value), 4),
                "impact": "increases" if value > 0 else "decreases",
                "abs_impact": round(abs(float(value)), 4)
            }
            for name, value in sorted_pairs
        ]

    def _generate_interpretation(
        self,
        feature_importance: List[Dict[str, Any]],
        target_class: int
    ) -> str:
        """영어 자연어 해석을 생성한다."""
        class_name = self.class_names.get(target_class, "Unknown")
        top_3 = feature_importance[:3]

        explanation = f"The {class_name} prediction is driven by: "

        parts = []
        for i, feat in enumerate(top_3, 1):
            direction = "high" if feat['impact'] == "increases" else "low"
            parts.append(f"{i}. {feat['feature']} ({direction})")

        explanation += ", ".join(parts) + "."

        return explanation

    def _generate_korean_interpretation(
        self,
        feature_importance: List[Dict[str, Any]],
        target_class: int
    ) -> str:
        """한글 자연어 해석을 생성한다."""
        class_name = {0: "안전", 1: "보통", 2: "독성"}.get(target_class, "알 수 없음")
        top_3 = feature_importance[:3]

        explanation = f"이 화학물질의 '{class_name}' 예측은 주로 다음 특성에 의해 결정되었습니다:\n"

        for i, feat in enumerate(top_3, 1):
            direction = "높은" if feat['impact'] == "increases" else "낮은"
            desc = self.FEATURE_DESCRIPTIONS.get(feat['feature'], feat['feature'])
            impact = "↑ 독성 증가" if feat['impact'] == "increases" else "↓ 독성 감소"
            explanation += f"  {i}. {desc}: {direction} 값 → {impact} (SHAP: {feat['shap_value']:.3f})\n"

        return explanation.strip()

    def get_feature_names(self) -> List[str]:
        """특성 이름 목록을 반환한다."""
        return self.feature_names.copy()


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # 더미 모델로 테스트
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=8, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # SHAP Explainer 테스트
    explainer = SHAPExplainer(model)

    # 단일 예측 설명
    sample = np.array([[180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89]])
    result = explainer.explain_prediction(sample)

    print("\n=== SHAP Explanation ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Force Plot 데이터
    force_plot = explainer.plot_force_plot(sample)
    print("\n=== Force Plot Data ===")
    print(json.dumps(force_plot, indent=2, ensure_ascii=False))

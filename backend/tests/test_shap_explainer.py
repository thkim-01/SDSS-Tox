"""Phase 3.2 SHAPExplainer 단위 테스트.

테스트 케이스:
1. Explainer 초기화 테스트
2. 예측 설명 테스트
3. Force Plot 데이터 테스트
4. Summary Statistics 테스트
5. 에러 처리 테스트

Author: DTO-DSS Team
Date: 2026-01-19
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.shap_explainer import SHAPExplainer, SHAPNotAvailableError


@pytest.fixture
def trained_model():
    """테스트용 학습된 RandomForest 모델을 생성한다."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_classes=3,
        random_state=42
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def explainer(trained_model) -> SHAPExplainer:
    """SHAPExplainer 인스턴스를 생성한다."""
    return SHAPExplainer(trained_model)


@pytest.fixture
def sample_features() -> np.ndarray:
    """샘플 특성 벡터를 반환한다."""
    return np.array([[180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89]])


class TestExplainerInitialization:
    """Explainer 초기화 테스트."""

    def test_explainer_creation(self, trained_model):
        """SHAPExplainer가 정상적으로 생성되는지 확인."""
        explainer = SHAPExplainer(trained_model)
        assert explainer is not None
        assert explainer.rf_model is not None
        assert explainer.explainer is not None

    def test_feature_names_count(self, explainer):
        """feature_names가 10개인지 확인."""
        assert len(explainer.feature_names) == 10
        assert explainer.feature_names[0] == "MW"
        assert explainer.feature_names[-1] == "logP"

    def test_class_names(self, explainer):
        """class_names가 올바르게 설정되어 있는지 확인."""
        assert explainer.class_names[0] == "Safe"
        assert explainer.class_names[1] == "Moderate"
        assert explainer.class_names[2] == "Toxic"

    def test_invalid_model_type(self):
        """RandomForest가 아닌 모델로 ValueError 발생 확인."""
        with pytest.raises(ValueError):
            SHAPExplainer({"not": "a model"})


class TestExplainPrediction:
    """예측 설명 테스트."""

    def test_explain_output_structure(self, explainer, sample_features):
        """explain_prediction 출력 구조 검증."""
        result = explainer.explain_prediction(sample_features)

        # 필수 키 확인
        assert "prediction" in result
        assert "target_class" in result
        assert "base_value" in result
        assert "shap_values" in result
        assert "feature_importance" in result
        assert "interpretation" in result
        assert "interpretation_ko" in result

    def test_shap_values_count(self, explainer, sample_features):
        """shap_values에 10개 특성이 포함되어 있는지 확인."""
        result = explainer.explain_prediction(sample_features)
        assert len(result['shap_values']) == 10
        assert "MW" in result['shap_values']
        assert "logKow" in result['shap_values']

    def test_feature_importance_sorted(self, explainer, sample_features):
        """feature_importance가 절대값 기준으로 정렬되어 있는지 확인."""
        result = explainer.explain_prediction(sample_features)
        importance = result['feature_importance']

        # 절대값 기준 내림차순 정렬 확인
        abs_values = [abs(f['shap_value']) for f in importance]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_feature_importance_structure(self, explainer, sample_features):
        """feature_importance 각 항목의 구조 확인."""
        result = explainer.explain_prediction(sample_features)
        importance = result['feature_importance']

        for item in importance:
            assert "feature" in item
            assert "shap_value" in item
            assert "impact" in item
            assert item["impact"] in ["increases", "decreases"]

    def test_interpretation_not_empty(self, explainer, sample_features):
        """interpretation이 비어있지 않은지 확인."""
        result = explainer.explain_prediction(sample_features)
        assert len(result['interpretation']) > 0
        assert len(result['interpretation_ko']) > 0

    def test_1d_input_accepted(self, explainer):
        """1D 배열 입력도 처리되는지 확인."""
        features_1d = np.array([180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89])
        result = explainer.explain_prediction(features_1d)
        assert "shap_values" in result


class TestForcePlot:
    """Force Plot 데이터 테스트."""

    def test_force_plot_structure(self, explainer, sample_features):
        """force_plot 출력 구조 검증."""
        result = explainer.plot_force_plot(sample_features)

        assert result["type"] == "force_plot"
        assert "base_value" in result
        assert "prediction" in result
        assert "features" in result

    def test_force_plot_features_count(self, explainer, sample_features):
        """force_plot features에 10개 항목이 있는지 확인."""
        result = explainer.plot_force_plot(sample_features)
        assert len(result["features"]) == 10

    def test_force_plot_feature_structure(self, explainer, sample_features):
        """force_plot 각 feature의 구조 확인."""
        result = explainer.plot_force_plot(sample_features)

        for feat in result["features"]:
            assert "name" in feat
            assert "value" in feat
            assert "shap_value" in feat
            assert "impact" in feat
            assert feat["impact"] in ["positive", "negative"]

    def test_positive_negative_separation(self, explainer, sample_features):
        """positive/negative features 분리 확인."""
        result = explainer.plot_force_plot(sample_features)

        assert "positive_features" in result
        assert "negative_features" in result

        # positive_features의 shap_value는 모두 양수
        for f in result["positive_features"]:
            assert f["shap_value"] > 0

        # negative_features의 shap_value는 모두 0 이하
        for f in result["negative_features"]:
            assert f["shap_value"] <= 0


class TestSummaryStatistics:
    """Summary Statistics 테스트."""

    def test_summary_statistics_structure(self, explainer):
        """summary_statistics 출력 구조 검증."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        assert "num_samples" in result
        assert "feature_importance" in result
        assert "plot_data" in result

    def test_summary_importance_ranked(self, explainer):
        """summary_statistics의 feature_importance가 순위화되어 있는지 확인."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        ranks = [f["rank"] for f in result["feature_importance"]]
        assert ranks == list(range(1, 11))

    def test_plot_data_structure(self, explainer):
        """plot_data 구조 확인."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        assert result["plot_data"]["type"] == "bar"
        assert len(result["plot_data"]["x"]) == 10
        assert len(result["plot_data"]["y"]) == 10


class TestErrorHandling:
    """에러 처리 테스트."""

    def test_wrong_feature_count(self, explainer):
        """잘못된 특성 개수로 ValueError 발생 확인."""
        wrong_features = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            explainer.explain_prediction(wrong_features)

    def test_3d_input_rejected(self, explainer):
        """3D 입력이 거부되는지 확인."""
        features_3d = np.random.randn(2, 3, 10)
        with pytest.raises(ValueError):
            explainer.explain_prediction(features_3d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

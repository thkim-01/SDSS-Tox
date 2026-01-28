"""Phase 3.2 SHAPExplainer  .

 :
1. Explainer  
2.   
3. Force Plot  
4. Summary Statistics 
5.   

Author: DTO-DSS Team
Date: 2026-01-19
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#   path 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.shap_explainer import SHAPExplainer, SHAPNotAvailableError


@pytest.fixture
def trained_model():
    """  RandomForest  ."""
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
    """SHAPExplainer  ."""
    return SHAPExplainer(trained_model)


@pytest.fixture
def sample_features() -> np.ndarray:
    """   ."""
    return np.array([[180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89]])


class TestExplainerInitialization:
    """Explainer  ."""

    def test_explainer_creation(self, trained_model):
        """SHAPExplainer   ."""
        explainer = SHAPExplainer(trained_model)
        assert explainer is not None
        assert explainer.rf_model is not None
        assert explainer.explainer is not None

    def test_feature_names_count(self, explainer):
        """feature_names 10 ."""
        assert len(explainer.feature_names) == 10
        assert explainer.feature_names[0] == "MW"
        assert explainer.feature_names[-1] == "logP"

    def test_class_names(self, explainer):
        """class_names    ."""
        assert explainer.class_names[0] == "Safe"
        assert explainer.class_names[1] == "Moderate"
        assert explainer.class_names[2] == "Toxic"

    def test_invalid_model_type(self):
        """RandomForest   ValueError  ."""
        with pytest.raises(ValueError):
            SHAPExplainer({"not": "a model"})


class TestExplainPrediction:
    """  ."""

    def test_explain_output_structure(self, explainer, sample_features):
        """explain_prediction   ."""
        result = explainer.explain_prediction(sample_features)

        #   
        assert "prediction" in result
        assert "target_class" in result
        assert "base_value" in result
        assert "shap_values" in result
        assert "feature_importance" in result
        assert "interpretation" in result
        assert "interpretation_ko" in result

    def test_shap_values_count(self, explainer, sample_features):
        """shap_values 10    ."""
        result = explainer.explain_prediction(sample_features)
        assert len(result['shap_values']) == 10
        assert "MW" in result['shap_values']
        assert "logKow" in result['shap_values']

    def test_feature_importance_sorted(self, explainer, sample_features):
        """feature_importance     ."""
        result = explainer.explain_prediction(sample_features)
        importance = result['feature_importance']

        #     
        abs_values = [abs(f['shap_value']) for f in importance]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_feature_importance_structure(self, explainer, sample_features):
        """feature_importance    ."""
        result = explainer.explain_prediction(sample_features)
        importance = result['feature_importance']

        for item in importance:
            assert "feature" in item
            assert "shap_value" in item
            assert "impact" in item
            assert item["impact"] in ["increases", "decreases"]

    def test_interpretation_not_empty(self, explainer, sample_features):
        """interpretation   ."""
        result = explainer.explain_prediction(sample_features)
        assert len(result['interpretation']) > 0
        assert len(result['interpretation_ko']) > 0

    def test_1d_input_accepted(self, explainer):
        """1D    ."""
        features_1d = np.array([180.16, 1.19, 1, 4, 3, 63.60, 1, 4, 13, 0.89])
        result = explainer.explain_prediction(features_1d)
        assert "shap_values" in result


class TestForcePlot:
    """Force Plot  ."""

    def test_force_plot_structure(self, explainer, sample_features):
        """force_plot   ."""
        result = explainer.plot_force_plot(sample_features)

        assert result["type"] == "force_plot"
        assert "base_value" in result
        assert "prediction" in result
        assert "features" in result

    def test_force_plot_features_count(self, explainer, sample_features):
        """force_plot features 10   ."""
        result = explainer.plot_force_plot(sample_features)
        assert len(result["features"]) == 10

    def test_force_plot_feature_structure(self, explainer, sample_features):
        """force_plot  feature  ."""
        result = explainer.plot_force_plot(sample_features)

        for feat in result["features"]:
            assert "name" in feat
            assert "value" in feat
            assert "shap_value" in feat
            assert "impact" in feat
            assert feat["impact"] in ["positive", "negative"]

    def test_positive_negative_separation(self, explainer, sample_features):
        """positive/negative features  ."""
        result = explainer.plot_force_plot(sample_features)

        assert "positive_features" in result
        assert "negative_features" in result

        # positive_features shap_value  
        for f in result["positive_features"]:
            assert f["shap_value"] > 0

        # negative_features shap_value  0 
        for f in result["negative_features"]:
            assert f["shap_value"] <= 0


class TestSummaryStatistics:
    """Summary Statistics ."""

    def test_summary_statistics_structure(self, explainer):
        """summary_statistics   ."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        assert "num_samples" in result
        assert "feature_importance" in result
        assert "plot_data" in result

    def test_summary_importance_ranked(self, explainer):
        """summary_statistics feature_importance   ."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        ranks = [f["rank"] for f in result["feature_importance"]]
        assert ranks == list(range(1, 11))

    def test_plot_data_structure(self, explainer):
        """plot_data  ."""
        test_set = np.random.randn(50, 10)
        result = explainer.summary_statistics(test_set)

        assert result["plot_data"]["type"] == "bar"
        assert len(result["plot_data"]["x"]) == 10
        assert len(result["plot_data"]["y"]) == 10


class TestErrorHandling:
    """  ."""

    def test_wrong_feature_count(self, explainer):
        """   ValueError  ."""
        wrong_features = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            explainer.explain_prediction(wrong_features)

    def test_3d_input_rejected(self, explainer):
        """3D   ."""
        features_3d = np.random.randn(2, 3, 10)
        with pytest.raises(ValueError):
            explainer.explain_prediction(features_3d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

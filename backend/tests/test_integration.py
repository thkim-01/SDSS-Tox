"""SDSS-Tox  .

Phase 6.1:    
- FastAPI  
-    
-   

Author: DTO-DSS Team
Date: 2026-01-20
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any

#   URL
BASE_URL = "http://localhost:8000"

#  
ASPIRIN_DESCRIPTORS = {
    "MW": 180.16,
    "logKow": 1.19,
    "HBD": 1,
    "HBA": 4,
    "nRotB": 3,
    "TPSA": 63.60,
    "Aromatic_Rings": 1,
    "Heteroatom_Count": 4,
    "Heavy_Atom_Count": 13,
    "logP": 0.89
}

TOXIC_DESCRIPTORS = {
    "MW": 450.5,
    "logKow": 5.2,
    "HBD": 0,
    "HBA": 2,
    "nRotB": 8,
    "TPSA": 25.0,
    "Aromatic_Rings": 4,
    "Heteroatom_Count": 3,
    "Heavy_Atom_Count": 35,
    "logP": 5.0
}


class TestHealthEndpoint:
    """  ."""
    
    def test_health_check(self):
        """   ."""
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "rf_predictor" in data
            assert "shap_explainer" in data
            assert "rule_engine" in data
            assert "ensemble_dss" in data
    
    def test_all_services_loaded(self):
        """   ."""
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/health")
            data = response.json()
            
            assert data["rf_predictor"] == "loaded"
            assert data["rule_engine"] == "loaded"
            assert data["simple_qsar"] == "loaded"
            assert data["read_across"] == "loaded"
            assert data["ensemble_dss"] == "loaded"


class TestPredictionEndpoint:
    """RF   ."""
    
    def test_predict_aspirin(self):
        """Aspirin    Safe  ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={
                    "chemical_id": "test_aspirin",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "class_name" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["chemical_id"] == "test_aspirin"
            assert data["class_name"] in ["Safe", "Moderate", "Toxic"]
    
    def test_predict_toxic_compound(self):
        """  ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={
                    "chemical_id": "test_toxic",
                    "descriptors": TOXIC_DESCRIPTORS
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            #  logP,    
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1
    
    def test_invalid_input(self):
        """    ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={
                    "chemical_id": "test",
                    "descriptors": {"MW": "invalid"}  #  
                }
            )
            
            assert response.status_code == 422  # Validation Error


class TestSHAPEndpoint:
    """SHAP   ."""
    
    def test_shap_explanation(self):
        """SHAP    ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/explainability/shap",
                json={
                    "chemical_id": "test_shap",
                    "descriptors": ASPIRIN_DESCRIPTORS,
                    "target_class": 2  # Toxic
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "feature_importance" in data
            assert "interpretation_ko" in data
            assert len(data["feature_importance"]) == 10  # 10 
    
    def test_shap_feature_importance_sorted(self):
        """    ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/explainability/shap",
                json={
                    "chemical_id": "test",
                    "descriptors": ASPIRIN_DESCRIPTORS,
                    "target_class": 0
                }
            )
            
            data = response.json()
            importance = data["feature_importance"]
            
            # abs_shap    
            abs_values = [f["abs_shap"] for f in importance]
            assert abs_values == sorted(abs_values, reverse=True)


class TestOntologyValidation:
    """   ."""
    
    def test_validate_endpoint(self):
        """   ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/ontology/validate",
                json={
                    "chemical_id": "test_onto",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "rf_prediction" in data
            assert "ontology_validation" in data
            assert "interpretation" in data
    
    def test_agreement_field(self):
        """Agreement     ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/ontology/validate",
                json={
                    "chemical_id": "test",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            data = response.json()
            agreement = data["ontology_validation"]["agreement"]
            
            assert agreement in ["AGREE", "DISAGREE", "INSUFFICIENT_DATA"]


class TestEnsembleAnalysis:
    """   ."""
    
    def test_ensemble_endpoint(self):
        """   ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={
                    "chemical_id": "test_ensemble",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "ensemble_results" in data
            assert "method_breakdown" in data
            assert "analysis_timestamp" in data
    
    def test_ensemble_has_four_methods(self):
        """ 4   ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={
                    "chemical_id": "test",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            data = response.json()
            methods = data["method_breakdown"]
            
            assert len(methods) == 4
            method_names = {m["method"] for m in methods}
            assert "RandomForest" in method_names
            assert "DTO Rules" in method_names
            assert "Simple QSAR" in method_names
            assert "Read-Across" in method_names
    
    def test_ensemble_weights_sum_to_one(self):
        """   1 ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={
                    "chemical_id": "test",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            data = response.json()
            methods = data["method_breakdown"]
            
            total_weight = sum(m["weight"] for m in methods)
            assert abs(total_weight - 1.0) < 0.01  #   
    
    def test_ensemble_recommendation(self):
        """    ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={
                    "chemical_id": "test",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            data = response.json()
            recommendation = data["ensemble_results"]["recommendation"]
            
            #  ACCEPT, REJECT, REVIEW NEEDED    
            assert any(word in recommendation for word in ["ACCEPT", "REJECT", "REVIEW"])
    
    def test_aspirin_is_safe(self):
        """Aspirin Safe  ."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={
                    "chemical_id": "aspirin",
                    "descriptors": ASPIRIN_DESCRIPTORS
                }
            )
            
            data = response.json()
            score = data["ensemble_results"]["score"]
            
            # Aspirin     
            assert score < 0.5


class TestEndToEnd:
    """  ."""
    
    def test_full_analysis_workflow(self):
        """   ."""
        with httpx.Client() as client:
            # 1. 
            health = client.get(f"{BASE_URL}/health").json()
            assert health["status"] == "healthy"
            
            # 2. RF 
            pred = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={"chemical_id": "e2e_test", "descriptors": ASPIRIN_DESCRIPTORS}
            ).json()
            rf_class = pred["class_name"]
            
            # 3. SHAP 
            target = pred["prediction_class"]
            shap = client.post(
                f"{BASE_URL}/explainability/shap",
                json={
                    "chemical_id": "e2e_test",
                    "descriptors": ASPIRIN_DESCRIPTORS,
                    "target_class": target
                }
            ).json()
            assert len(shap["feature_importance"]) == 10
            
            # 4.  
            onto = client.post(
                f"{BASE_URL}/ontology/validate",
                json={"chemical_id": "e2e_test", "descriptors": ASPIRIN_DESCRIPTORS}
            ).json()
            assert "agreement" in onto["ontology_validation"]
            
            # 5.  
            ensemble = client.post(
                f"{BASE_URL}/analysis/ensemble",
                json={"chemical_id": "e2e_test", "descriptors": ASPIRIN_DESCRIPTORS}
            ).json()
            assert "recommendation" in ensemble["ensemble_results"]
            
            print(f"\nE2E Test Passed!")
            print(f"   RF Prediction: {rf_class}")
            print(f"   Ensemble Score: {ensemble['ensemble_results']['score']:.2%}")
            print(f"   Recommendation: {ensemble['ensemble_results']['recommendation']}")


if __name__ == "__main__":
    #    
    print("=" * 50)
    print("DTO-DSS Integration Tests")
    print("=" * 50)
    
    pytest.main([__file__, "-v", "--tb=short"])

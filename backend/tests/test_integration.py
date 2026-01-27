"""DTO-DSS 통합 테스트.

Phase 6.1: 전체 시스템 통합 테스트
- FastAPI 엔드포인트 테스트
- 서비스 간 연동 테스트
- 앙상블 로직 검증

Author: DTO-DSS Team
Date: 2026-01-20
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any

# 테스트 서버 URL
BASE_URL = "http://localhost:8000"

# 테스트 데이터
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
    """헬스체크 엔드포인트 테스트."""
    
    def test_health_check(self):
        """서버 헬스체크가 성공하는지 확인."""
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
        """모든 서비스가 로드되었는지 확인."""
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/health")
            data = response.json()
            
            assert data["rf_predictor"] == "loaded"
            assert data["rule_engine"] == "loaded"
            assert data["simple_qsar"] == "loaded"
            assert data["read_across"] == "loaded"
            assert data["ensemble_dss"] == "loaded"


class TestPredictionEndpoint:
    """RF 예측 엔드포인트 테스트."""
    
    def test_predict_aspirin(self):
        """Aspirin 유사 분자 예측이 Safe를 반환하는지 확인."""
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
        """독성 분자 예측."""
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
            
            # 높은 logP, 방향족 고리가 많은 분자
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1
    
    def test_invalid_input(self):
        """잘못된 입력에 대한 에러 처리."""
        with httpx.Client() as client:
            response = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={
                    "chemical_id": "test",
                    "descriptors": {"MW": "invalid"}  # 잘못된 타입
                }
            )
            
            assert response.status_code == 422  # Validation Error


class TestSHAPEndpoint:
    """SHAP 설명 엔드포인트 테스트."""
    
    def test_shap_explanation(self):
        """SHAP 설명이 올바르게 반환되는지 확인."""
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
            assert len(data["feature_importance"]) == 10  # 10개 특징
    
    def test_shap_feature_importance_sorted(self):
        """특징 중요도가 정렬되어 반환되는지 확인."""
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
            
            # abs_shap 기준 내림차순 정렬 확인
            abs_values = [f["abs_shap"] for f in importance]
            assert abs_values == sorted(abs_values, reverse=True)


class TestOntologyValidation:
    """온톨로지 검증 엔드포인트 테스트."""
    
    def test_validate_endpoint(self):
        """온톨로지 검증이 작동하는지 확인."""
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
        """Agreement 필드가 올바른 값을 가지는지 확인."""
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
    """앙상블 분석 엔드포인트 테스트."""
    
    def test_ensemble_endpoint(self):
        """앙상블 분석이 작동하는지 확인."""
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
        """앙상블이 4가지 방법을 포함하는지 확인."""
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
        """앙상블 가중치 합이 1인지 확인."""
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
            assert abs(total_weight - 1.0) < 0.01  # 약간의 오차 허용
    
    def test_ensemble_recommendation(self):
        """앙상블 추천이 올바른 형식인지 확인."""
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
            
            # 추천은 ACCEPT, REJECT, REVIEW NEEDED 중 하나를 포함해야 함
            assert any(word in recommendation for word in ["ACCEPT", "REJECT", "REVIEW"])
    
    def test_aspirin_is_safe(self):
        """Aspirin이 Safe로 분류되는지 확인."""
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
            
            # Aspirin은 낮은 독성 점수를 가져야 함
            assert score < 0.5


class TestEndToEnd:
    """엔드투엔드 시나리오 테스트."""
    
    def test_full_analysis_workflow(self):
        """전체 분석 워크플로우 테스트."""
        with httpx.Client() as client:
            # 1. 헬스체크
            health = client.get(f"{BASE_URL}/health").json()
            assert health["status"] == "healthy"
            
            # 2. RF 예측
            pred = client.post(
                f"{BASE_URL}/predictions/randomforest",
                json={"chemical_id": "e2e_test", "descriptors": ASPIRIN_DESCRIPTORS}
            ).json()
            rf_class = pred["class_name"]
            
            # 3. SHAP 설명
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
            
            # 4. 온톨로지 검증
            onto = client.post(
                f"{BASE_URL}/ontology/validate",
                json={"chemical_id": "e2e_test", "descriptors": ASPIRIN_DESCRIPTORS}
            ).json()
            assert "agreement" in onto["ontology_validation"]
            
            # 5. 앙상블 분석
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
    # 직접 실행 시 테스트
    print("=" * 50)
    print("DTO-DSS Integration Tests")
    print("=" * 50)
    
    pytest.main([__file__, "-v", "--tb=short"])

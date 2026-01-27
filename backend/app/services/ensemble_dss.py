"""Ensemble 의사결정 지원 시스템.

Phase 5.1: 4가지 예측 방법 통합
- Read-Across (35%)
- Simple QSAR (15%)
- RandomForest (25%)
- DTO Rules (25%)

앙상블 점수, 신뢰도, 추천을 생성합니다.

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """의사결정 추천."""
    STRONG_ACCEPT = "STRONG ACCEPT"
    ACCEPT = "ACCEPT"
    REVIEW_NEEDED = "REVIEW NEEDED"
    REJECT = "REJECT"
    STRONG_REJECT = "STRONG REJECT"


@dataclass
class MethodResult:
    """개별 예측 방법 결과."""
    method: str
    score: float  # 0.0 (Safe) - 1.0 (Toxic)
    confidence: float  # 0.0 - 1.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleResult:
    """앙상블 결과."""
    chemical_id: Optional[str]
    ensemble_score: float
    ensemble_confidence: float
    ensemble_class: str
    method_agreement: float
    recommendation: str
    method_breakdown: List[Dict[str, Any]]
    detailed_reasoning: str
    analysis_timestamp: str


class EnsembleDSS:
    """4가지 예측 방법을 결합하는 앙상블 의사결정 지원 시스템.

    각 예측 방법에 가중치를 부여하고, 최종 점수와 추천을 생성합니다.

    Weights:
        - Read-Across: 35% (실제 유사 물질 데이터 기반)
        - Simple QSAR: 15% (기본 규칙 기반)
        - RandomForest: 25% (ML 모델)
        - DTO Rules: 25% (온톨로지 규칙)

    Example:
        >>> dss = EnsembleDSS()
        >>> rf_result = MethodResult("RandomForest", 0.82, 0.88)
        >>> dto_result = MethodResult("DTO Rules", 0.89, 0.92)
        >>> result = dss.combine_predictions(rf=rf_result, dto=dto_result)
        >>> print(result.recommendation)
    """

    # 기본 가중치
    DEFAULT_WEIGHTS = {
        "Read-Across": 0.35,
        "Simple QSAR": 0.15,
        "RandomForest": 0.25,
        "DTO Rules": 0.25
    }

    # 클래스 분류 임계값
    TOXIC_THRESHOLD = 0.6
    SAFE_THRESHOLD = 0.4

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """EnsembleDSS를 초기화한다.

        Args:
            weights: 커스텀 가중치 딕셔너리 (선택사항).
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        logger.info(f"EnsembleDSS initialized with weights: {self.weights}")

    def combine_predictions(
        self,
        chemical_id: Optional[str] = None,
        readacross: Optional[MethodResult] = None,
        qsar: Optional[MethodResult] = None,
        rf: Optional[MethodResult] = None,
        dto: Optional[MethodResult] = None
    ) -> EnsembleResult:
        """4가지 예측 방법을 결합하여 최종 결과를 생성한다.

        Args:
            chemical_id: 화학물질 ID.
            readacross: Read-Across 결과.
            qsar: Simple QSAR 결과.
            rf: RandomForest 결과.
            dto: DTO Rules 결과.

        Returns:
            앙상블 결과.
        """
        # 가용한 방법 수집
        methods = []
        
        if readacross:
            methods.append((readacross, self.weights.get("Read-Across", 0.35)))
        if qsar:
            methods.append((qsar, self.weights.get("Simple QSAR", 0.15)))
        if rf:
            methods.append((rf, self.weights.get("RandomForest", 0.25)))
        if dto:
            methods.append((dto, self.weights.get("DTO Rules", 0.25)))

        if not methods:
            raise ValueError("At least one prediction method result is required")

        logger.info(f"Combining {len(methods)} prediction methods for {chemical_id}")

        # 가중치 정규화
        total_weight = sum(w for _, w in methods)
        normalized_methods = [(m, w / total_weight) for m, w in methods]

        # 가중 평균 점수 계산
        weighted_score = sum(m.score * w for m, w in normalized_methods)

        # 신뢰도 계산 (보수적: 최소값)
        min_confidence = min(m.confidence for m, _ in methods)
        
        # 가중 평균 신뢰도
        weighted_confidence = sum(m.confidence * w for m, w in normalized_methods)

        # 방법 간 합의도 계산
        predictions = [m.score > 0.5 for m, _ in methods]
        method_agreement = self._calculate_agreement(predictions)

        # 최종 분류
        ensemble_class = self._classify(weighted_score)

        # 최종 추천
        recommendation = self._make_recommendation(
            weighted_score,
            min_confidence,
            method_agreement
        )

        # 방법별 상세 분석
        method_breakdown = [
            {
                "method": m.method,
                "score": round(m.score, 3),
                "confidence": round(m.confidence, 3),
                "weight": round(w, 3),
                "weighted_contribution": round(m.score * w, 3),
                "prediction": "Toxic" if m.score > 0.5 else "Safe"
            }
            for m, w in normalized_methods
        ]

        # 상세 추론 생성
        detailed_reasoning = self._generate_reasoning(
            chemical_id,
            normalized_methods,
            weighted_score,
            ensemble_class,
            method_agreement
        )

        return EnsembleResult(
            chemical_id=chemical_id,
            ensemble_score=round(weighted_score, 4),
            ensemble_confidence=round(weighted_confidence, 4),
            ensemble_class=ensemble_class,
            method_agreement=round(method_agreement, 4),
            recommendation=recommendation,
            method_breakdown=method_breakdown,
            detailed_reasoning=detailed_reasoning,
            analysis_timestamp=datetime.now().isoformat()
        )

    def combine_from_rf_and_dto(
        self,
        chemical_id: Optional[str],
        rf_prediction: Dict[str, Any],
        dto_validation: Dict[str, Any]
    ) -> EnsembleResult:
        """RF 예측과 DTO 검증 결과를 결합한다.

        Args:
            chemical_id: 화학물질 ID.
            rf_prediction: RFPredictor 결과.
            dto_validation: DTORuleEngine 검증 결과.

        Returns:
            앙상블 결과.
        """
        # RF 결과 변환
        rf_class = rf_prediction.get("class_name", "Unknown")
        rf_confidence = rf_prediction.get("confidence", 0.5)
        
        # class_name을 score로 변환
        rf_score = self._class_to_score(rf_class)
        
        rf_result = MethodResult(
            method="RandomForest",
            score=rf_score,
            confidence=rf_confidence,
            details=rf_prediction
        )

        # DTO 결과 변환
        dto_agreement = dto_validation.get("agreement", "INSUFFICIENT_DATA")
        dto_rule_confidence = dto_validation.get("rule_confidence_score", 0.5)
        
        # agreement를 score로 변환
        if dto_agreement == "AGREE":
            dto_score = rf_score  # RF와 동일
        elif dto_agreement == "DISAGREE":
            dto_score = 1.0 - rf_score  # RF와 반대
        else:
            dto_score = 0.5  # 불확실

        dto_result = MethodResult(
            method="DTO Rules",
            score=dto_score,
            confidence=dto_rule_confidence,
            details=dto_validation
        )

        return self.combine_predictions(
            chemical_id=chemical_id,
            rf=rf_result,
            dto=dto_result
        )

    def _class_to_score(self, class_name: str) -> float:
        """클래스 이름을 점수로 변환한다."""
        class_scores = {
            "Safe": 0.1,
            "Moderate": 0.5,
            "Toxic": 0.9,
            "Unknown": 0.5
        }
        return class_scores.get(class_name, 0.5)

    def _calculate_agreement(self, predictions: List[bool]) -> float:
        """방법 간 합의도를 계산한다."""
        if not predictions:
            return 0.0

        # 다수결 기준 합의도
        toxic_count = sum(predictions)
        majority = max(toxic_count, len(predictions) - toxic_count)
        
        return majority / len(predictions)

    def _classify(self, score: float) -> str:
        """점수를 기반으로 분류한다."""
        if score >= self.TOXIC_THRESHOLD:
            return "Toxic"
        elif score <= self.SAFE_THRESHOLD:
            return "Safe"
        else:
            return "Moderate"

    def _make_recommendation(
        self,
        score: float,
        confidence: float,
        agreement: float
    ) -> str:
        """최종 추천을 생성한다."""
        # Strong Reject: 높은 독성 + 높은 신뢰도 + 높은 합의
        if score > 0.7 and confidence > 0.8 and agreement > 0.8:
            return Recommendation.STRONG_REJECT.value + " (high confidence, high agreement)"

        # Reject: 높은 독성
        elif score > 0.7:
            return Recommendation.REJECT.value + " (moderate to high confidence)"

        # Strong Accept: 낮은 독성 + 높은 신뢰도 + 높은 합의
        elif score < 0.3 and confidence > 0.8 and agreement > 0.8:
            return Recommendation.STRONG_ACCEPT.value + " (high confidence, high agreement)"

        # Accept: 낮은 독성
        elif score < 0.3:
            return Recommendation.ACCEPT.value + " (moderate to high confidence)"

        # Review Needed: 중간 범위
        else:
            return Recommendation.REVIEW_NEEDED.value + " (moderate risk, requires expert review)"

    def _generate_reasoning(
        self,
        chemical_id: Optional[str],
        methods: List[tuple],
        weighted_score: float,
        ensemble_class: str,
        agreement: float
    ) -> str:
        """상세 추론을 생성한다."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("           앙상블 DSS 분석 보고서")
        lines.append("=" * 60)
        lines.append("")
        
        if chemical_id:
            lines.append(f"화학물질 ID: {chemical_id}")
        
        lines.append(f"분석된 방법 수: {len(methods)}개")
        lines.append("")
        
        # 방법별 결과
        lines.append("방법별 결과:")
        for m, w in methods:
            pred = "독성" if m.score > 0.5 else "안전"
            lines.append(f"   - {m.method}: {pred} ({m.score:.0%}) - 신뢰도: {m.confidence:.0%}")
        
        lines.append("")
        
        # 앙상블 결과
        lines.append("━" * 40)
        lines.append(f"앙상블 점수: {weighted_score:.1%}")
        lines.append(f"최종 분류: {ensemble_class}")
        lines.append(f"방법 간 합의도: {agreement:.0%}")
        lines.append("━" * 40)
        
        lines.append("")
        
        # 해석
        if agreement > 0.8:
            lines.append("모든 방법이 일치하는 결과를 보여 신뢰도가 높습니다.")
        elif agreement > 0.5:
            lines.append("방법 간 부분적 불일치가 있어 추가 검토가 필요합니다.")
        else:
            lines.append("방법 간 큰 불일치가 있어 전문가 검토가 필수적입니다.")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)

    def to_dict(self, result: EnsembleResult) -> Dict[str, Any]:
        """EnsembleResult를 딕셔너리로 변환한다."""
        return {
            "chemical_id": result.chemical_id,
            "ensemble_results": {
                "score": result.ensemble_score,
                "confidence": result.ensemble_confidence,
                "class": result.ensemble_class,
                "method_agreement": result.method_agreement,
                "recommendation": result.recommendation
            },
            "method_breakdown": result.method_breakdown,
            "detailed_reasoning": result.detailed_reasoning,
            "analysis_timestamp": result.analysis_timestamp
        }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # 테스트
    dss = EnsembleDSS()

    # 시뮬레이션 결과
    rf_result = MethodResult("RandomForest", 0.82, 0.88)
    dto_result = MethodResult("DTO Rules", 0.89, 0.92)
    readacross_result = MethodResult("Read-Across", 0.75, 0.85)
    qsar_result = MethodResult("Simple QSAR", 0.68, 0.70)

    result = dss.combine_predictions(
        chemical_id="drug_candidate_001",
        readacross=readacross_result,
        qsar=qsar_result,
        rf=rf_result,
        dto=dto_result
    )

    print(result.detailed_reasoning)
    print(f"\nRecommendation: {result.recommendation}")
    print(f"Ensemble Score: {result.ensemble_score:.2%}")

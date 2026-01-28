"""Ensemble   .

Phase 5.1: 4   
- Read-Across (35%)
- Simple QSAR (15%)
- RandomForest (25%)
- DTO Rules (25%)

 , ,  .

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

#  
logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """ ."""
    STRONG_ACCEPT = "STRONG ACCEPT"
    ACCEPT = "ACCEPT"
    REVIEW_NEEDED = "REVIEW NEEDED"
    REJECT = "REJECT"
    STRONG_REJECT = "STRONG REJECT"


@dataclass
class MethodResult:
    """   ."""
    method: str
    score: float  # 0.0 (Safe) - 1.0 (Toxic)
    confidence: float  # 0.0 - 1.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleResult:
    """ ."""
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
    """4       .

        ,    .

    Weights:
        - Read-Across: 35% (    )
        - Simple QSAR: 15% (  )
        - RandomForest: 25% (ML )
        - DTO Rules: 25% ( )

    Example:
        >>> dss = EnsembleDSS()
        >>> rf_result = MethodResult("RandomForest", 0.82, 0.88)
        >>> dto_result = MethodResult("DTO Rules", 0.89, 0.92)
        >>> result = dss.combine_predictions(rf=rf_result, dto=dto_result)
        >>> print(result.recommendation)
    """

    #  
    DEFAULT_WEIGHTS = {
        "Read-Across": 0.35,
        "Simple QSAR": 0.15,
        "RandomForest": 0.25,
        "DTO Rules": 0.25
    }

    #   
    TOXIC_THRESHOLD = 0.6
    SAFE_THRESHOLD = 0.4

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """EnsembleDSS .

        Args:
            weights:    ().
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
        """4      .

        Args:
            chemical_id:  ID.
            readacross: Read-Across .
            qsar: Simple QSAR .
            rf: RandomForest .
            dto: DTO Rules .

        Returns:
             .
        """
        #   
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

        #  
        total_weight = sum(w for _, w in methods)
        normalized_methods = [(m, w / total_weight) for m, w in methods]

        #    
        weighted_score = sum(m.score * w for m, w in normalized_methods)

        #   (: )
        min_confidence = min(m.confidence for m, _ in methods)
        
        #   
        weighted_confidence = sum(m.confidence * w for m, w in normalized_methods)

        #    
        predictions = [m.score > 0.5 for m, _ in methods]
        method_agreement = self._calculate_agreement(predictions)

        #  
        ensemble_class = self._classify(weighted_score)

        #  
        recommendation = self._make_recommendation(
            weighted_score,
            min_confidence,
            method_agreement
        )

        #   
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

        #   
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
        """RF  DTO   .

        Args:
            chemical_id:  ID.
            rf_prediction: RFPredictor .
            dto_validation: DTORuleEngine  .

        Returns:
             .
        """
        # RF  
        rf_class = rf_prediction.get("class_name", "Unknown")
        rf_confidence = rf_prediction.get("confidence", 0.5)
        
        # class_name score 
        rf_score = self._class_to_score(rf_class)
        
        rf_result = MethodResult(
            method="RandomForest",
            score=rf_score,
            confidence=rf_confidence,
            details=rf_prediction
        )

        # DTO  
        dto_agreement = dto_validation.get("agreement", "INSUFFICIENT_DATA")
        dto_rule_confidence = dto_validation.get("rule_confidence_score", 0.5)
        
        # agreement score 
        if dto_agreement == "AGREE":
            dto_score = rf_score  # RF 
        elif dto_agreement == "DISAGREE":
            dto_score = 1.0 - rf_score  # RF 
        else:
            dto_score = 0.5  # 

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
        """   ."""
        class_scores = {
            "Safe": 0.1,
            "Moderate": 0.5,
            "Toxic": 0.9,
            "Unknown": 0.5
        }
        return class_scores.get(class_name, 0.5)

    def _calculate_agreement(self, predictions: List[bool]) -> float:
        """   ."""
        if not predictions:
            return 0.0

        #   
        toxic_count = sum(predictions)
        majority = max(toxic_count, len(predictions) - toxic_count)
        
        return majority / len(predictions)

    def _classify(self, score: float) -> str:
        """  ."""
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
        """  ."""
        # Strong Reject:   +   +  
        if score > 0.7 and confidence > 0.8 and agreement > 0.8:
            return Recommendation.STRONG_REJECT.value + " (high confidence, high agreement)"

        # Reject:  
        elif score > 0.7:
            return Recommendation.REJECT.value + " (moderate to high confidence)"

        # Strong Accept:   +   +  
        elif score < 0.3 and confidence > 0.8 and agreement > 0.8:
            return Recommendation.STRONG_ACCEPT.value + " (high confidence, high agreement)"

        # Accept:  
        elif score < 0.3:
            return Recommendation.ACCEPT.value + " (moderate to high confidence)"

        # Review Needed:  
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
        """  ."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("            DSS  ")
        lines.append("=" * 60)
        lines.append("")
        
        if chemical_id:
            lines.append(f" ID: {chemical_id}")
        
        lines.append(f"  : {len(methods)}")
        lines.append("")
        
        #  
        lines.append(" :")
        for m, w in methods:
            pred = "" if m.score > 0.5 else ""
            lines.append(f"   - {m.method}: {pred} ({m.score:.0%}) - : {m.confidence:.0%}")
        
        lines.append("")
        
        #  
        lines.append("" * 40)
        lines.append(f" : {weighted_score:.1%}")
        lines.append(f" : {ensemble_class}")
        lines.append(f"  : {agreement:.0%}")
        lines.append("" * 40)
        
        lines.append("")
        
        # 
        if agreement > 0.8:
            lines.append("      .")
        elif agreement > 0.5:
            lines.append("       .")
        else:
            lines.append("       .")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)

    def to_dict(self, result: EnsembleResult) -> Dict[str, Any]:
        """EnsembleResult  ."""
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

    # 
    dss = EnsembleDSS()

    #  
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

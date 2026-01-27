"""Simple QSAR service for descriptor calculation and heuristic scoring.

This module provides a lightweight, RDKit-backed descriptor calculator
and a simple scoring heuristic for prototyping QSAR functionality.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RDKitNotAvailable(RuntimeError):
    pass


class SimpleQSAR:
    """Lightweight QSAR utility.

    Methods:
      - compute_descriptors(smiles) -> Dict[str, float]
      - predict(descriptors) -> Dict[str, Any]
    """

    def __init__(self) -> None:
        # Lazy RDKit import
        try:
            from rdkit import Chem  # type: ignore
            # import to verify availability
            _ = Chem
            self._rdkit_available = True
        except Exception:
            self._rdkit_available = False
            logger.warning("RDKit not available: SimpleQSAR will be limited")

    def _require_rdkit(self) -> None:
        if not self._rdkit_available:
            raise RDKitNotAvailable("RDKit is required for descriptor calculation")

    def compute_descriptors(self, smiles: str) -> Dict[str, float]:
        """Compute a set of descriptors from a SMILES string.

        Returns a dictionary containing the standard 10-feature vector used
        by RFPredictor plus any additional helpful descriptors.
        """
        self._require_rdkit()
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        desc = {
            "MW": float(Descriptors.MolWt(mol)),
            "logKow": float(Descriptors.MolLogP(mol)),
            "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
            "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
            "nRotB": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "Aromatic_Rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "Heteroatom_Count": int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))),
            "Heavy_Atom_Count": int(mol.GetNumHeavyAtoms()),
            "logP": float(Descriptors.MolLogP(mol)),
        }

        return desc

    def predict(self, descriptors: Dict[str, float]) -> Dict[str, Any]:
        """Return a simple heuristic QSAR score from descriptor dict.

        Heuristic: score in [0,1] computed from MW, logP, TPSA. Confidence
        is a simple measure based on how well descriptors fall into known ranges.
        """
        # Validate keys
        required = ["MW", "logP", "TPSA"]
        if not all(k in descriptors for k in required):
            raise ValueError(f"Descriptors missing required keys: {required}")

        mw = float(descriptors["MW"])
        logp = float(descriptors.get("logP", descriptors.get("logKow", 0.0)))
        tpsa = float(descriptors["TPSA"])

        # Simple normalized components
        mw_score = min(max((mw - 100.0) / 500.0, 0.0), 1.0)
        logp_score = min(max((logp + 2.0) / 7.0, 0.0), 1.0)
        tpsa_score = 1.0 - min(max(tpsa / 200.0, 0.0), 1.0)

        score = 0.45 * mw_score + 0.35 * logp_score + 0.20 * tpsa_score
        score = float(min(max(score, 0.0), 1.0))

        # Confidence heuristic: how many descriptors are within 'typical' ranges
        conf = 0.0
        conf += 1.0 if 100.0 <= mw <= 500.0 else 0.0
        conf += 1.0 if -2.0 <= logp <= 5.0 else 0.0
        conf += 1.0 if 0.0 <= tpsa <= 140.0 else 0.0
        confidence = float(conf / 3.0)

        label = "Toxic" if score > 0.6 else ("Safe" if score < 0.3 else "Moderate")

        return {
            "score": score,
            "confidence": confidence,
            "label": label,
            "details": {
                "components": {"mw_score": mw_score, "logp_score": logp_score, "tpsa_score": tpsa_score}
            }
        }


__all__ = ["SimpleQSAR", "RDKitNotAvailable"]
"""Simple QSAR 예측 서비스.

Phase 5.1: 4가지 예측 방법 중 하나
- 물리화학적 특성(Descriptors) 기반의 간단한 규칙 적용
- Lipinski Rule of 5 등 활용

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from typing import Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)


class SimpleQSAR:
    """간단한 QSAR 규칙 기반 예측기.
    
    복잡한 ML 모델이 아닌, 잘 알려진 의약화학 규칙들을 사용하여
    독성 가능성을 평가합니다.
    """

    def __init__(self):
        logger.info("SimpleQSAR initialized")

    def predict(self, descriptors: Dict[str, float]) -> Dict[str, Any]:
        """분자 기술자를 기반으로 독성 점수를 예측합니다.
        
        Args:
            descriptors: 분자 기술자 딕셔너리 (MW, logP, HBD, HBA 등)
            
        Returns:
            예측 결과 (score, confidence, details)
        """
        score = 0.0
        risk_factors = []
        
        # 1. Lipinski Rule of 5 위반 (약물성 저하 -> 독성/부작용 연관 가능성)
        lipinski_violations = 0
        if descriptors.get("MW", 0) > 500:
            lipinski_violations += 1
            risk_factors.append("MW > 500")
        if descriptors.get("logP", 0) > 5:
            lipinski_violations += 1
            risk_factors.append("logP > 5")
        if descriptors.get("HBD", 0) > 5:
            lipinski_violations += 1
            risk_factors.append("HBD > 5")
        if descriptors.get("HBA", 0) > 10:
            lipinski_violations += 1
            risk_factors.append("HBA > 10")
            
        # 위반이 많을수록 점수 증가 (간접적 독성 지표)
        score += lipinski_violations * 0.1
        
        # 2. QSAR 독성 경고 구조 (Structural Alerts) - 간단한 수치 기준
        # TPSA가 매우 낮으면(<75) 세포막 투과성이 높아 독성 위험 증가 가능
        if descriptors.get("TPSA", 100) < 75:
            score += 0.2
            risk_factors.append("Low TPSA (<75)")
            
        # 방향족 고리가 많으면(>3) 대사 안정성 문제 및 독성 가능성
        if descriptors.get("Aromatic_Rings", 0) > 3:
            score += 0.3
            risk_factors.append("High Aromatic Rings (>3)")
            
        # logKow가 매우 높으면(>4) 생체 축적 가능성
        if descriptors.get("logKow", 0) > 4.0:
            score += 0.2
            risk_factors.append("High logKow (>4.0)")

        # 점수 정규화 (0.0 ~ 1.0)
        final_score = min(1.0, score)
        
        # 신뢰도: 적용된 규칙의 수에 비례 (단순화)
        confidence = 0.6 + (len(risk_factors) * 0.05)
        confidence = min(0.9, confidence)
        
        prediction_class = "Toxic" if final_score > 0.5 else "Safe"
        
        return {
            "method": "Simple QSAR",
            "score": round(final_score, 3),
            "confidence": round(confidence, 2),
            "class": prediction_class,
            "details": {
                "risk_factors": risk_factors,
                "lipinski_violations": lipinski_violations
            }
        }

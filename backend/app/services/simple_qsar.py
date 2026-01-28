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
"""Simple QSAR  .

Phase 5.1: 4    
-  (Descriptors)    
- Lipinski Rule of 5  

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from typing import Dict, Any

#  
logger = logging.getLogger(__name__)


class SimpleQSAR:
    """ QSAR   .
    
     ML  ,     
      .
    """

    def __init__(self):
        logger.info("SimpleQSAR initialized")

    def predict(self, descriptors: Dict[str, float]) -> Dict[str, Any]:
        """     .
        
        Args:
            descriptors:    (MW, logP, HBD, HBA )
            
        Returns:
              (score, confidence, details)
        """
        score = 0.0
        risk_factors = []
        
        # 1. Lipinski Rule of 5  (  -> /  )
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
            
        #     (  )
        score += lipinski_violations * 0.1
        
        # 2. QSAR    (Structural Alerts) -   
        # TPSA  (<75)       
        if descriptors.get("TPSA", 100) < 75:
            score += 0.2
            risk_factors.append("Low TPSA (<75)")
            
        #   (>3)      
        if descriptors.get("Aromatic_Rings", 0) > 3:
            score += 0.3
            risk_factors.append("High Aromatic Rings (>3)")
            
        # logKow  (>4)   
        if descriptors.get("logKow", 0) > 4.0:
            score += 0.2
            risk_factors.append("High logKow (>4.0)")

        #   (0.0 ~ 1.0)
        final_score = min(1.0, score)
        
        # :     ()
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

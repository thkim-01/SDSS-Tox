"""Drug Target Ontology  .

Phase 4.2: DTO Rule Engine
-      
- RandomForest   
-    /

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

#  
logger = logging.getLogger(__name__)


class ToxicityLevel(Enum):
    """ ."""
    SAFE = 0
    MODERATE = 1
    TOXIC = 2
    UNKNOWN = -1


@dataclass
class Rule:
    """ ."""
    name: str
    description: str
    antecedents: List[Tuple[str, str, str]]  # (predicate, subject, object) 
    consequent: Tuple[str, Any]  # (property, value)
    confidence: float
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """ ."""
        return {
            "name": self.name,
            "description": self.description,
            "antecedents": [
                {"predicate": a[0], "subject": a[1], "object": a[2]}
                for a in self.antecedents
            ],
            "consequent": {"property": self.consequent[0], "value": self.consequent[1]},
            "confidence": self.confidence,
            "category": self.category
        }


@dataclass
class RuleMatchResult:
    """  ."""
    rule_name: str
    matched: bool
    antecedents_met: List[str]
    conclusion: str
    confidence: float
    relevance_to_rf: str


@dataclass
class ValidationResult:
    """ ."""
    chemical_id: Optional[str]
    rf_prediction: str
    rf_confidence: float
    matched_rules: List[RuleMatchResult]
    rule_count: int
    rule_confidence_score: float
    combined_confidence: float
    agreement: str  # "AGREE", "DISAGREE", "INSUFFICIENT_DATA"
    interpretation: str


class DTORuleEngine:
    """Drug Target Ontology   .

        RandomForest  .

    Attributes:
        rules:   .
        dto_parser: DTOParser  ().

    Example:
        >>> engine = DTORuleEngine()
        >>> rf_pred = {"class_name": "Toxic", "confidence": 0.82}
        >>> result = engine.validate_prediction("aspirin", rf_pred, descriptors)
        >>> print(result.agreement)  # "AGREE" or "DISAGREE"
    """

    #     
    DESCRIPTOR_RULES: List[Rule] = [
        Rule(
            name="high_logkow_toxicity",
            description="High logKow (>3.0) increases cell membrane permeability and toxicity risk",
            antecedents=[("logKow", "?chemical", ">3.0")],
            consequent=("toxicity_risk", "High"),
            confidence=0.85,
            category="descriptor"
        ),
        Rule(
            name="high_molecular_weight",
            description="High MW (>500) reduces bioavailability but may indicate toxicity",
            antecedents=[("MW", "?chemical", ">500")],
            consequent=("bioavailability", "Low"),
            confidence=0.75,
            category="descriptor"
        ),
        Rule(
            name="lipinski_violation_hbd",
            description="HBD > 5 violates Lipinski Rule (Toxicity Risk)",
            antecedents=[("HBD", "?chemical", ">5")],
            consequent=("lipinski_violation", True),
            confidence=0.80,
            category="descriptor"
        ),
        Rule(
            name="lipinski_violation_hba",
            description="HBA > 10 violates Lipinski Rule",
            antecedents=[("HBA", "?chemical", ">10")],
            consequent=("lipinski_violation", True),
            confidence=0.80,
            category="descriptor"
        ),
        Rule(
            name="aromatic_toxicity",
            description="Aromatic Rings > 3 increases metabolic activation toxicity risk",
            antecedents=[("Aromatic_Rings", "?chemical", ">3")],
            consequent=("metabolic_toxicity_risk", "Elevated"),
            confidence=0.70,
            category="descriptor"
        ),
        Rule(
            name="low_tpsa_permeability",
            description="Low TPSA (<40) indicates high membrane permeability",
            antecedents=[("TPSA", "?chemical", "<40")],
            consequent=("membrane_permeability", "High"),
            confidence=0.75,
            category="descriptor"
        ),
        Rule(
            name="high_rotatable_bonds",
            description="Rotatable Bonds > 10 indicates high molecular flexibility",
            antecedents=[("nRotB", "?chemical", ">10")],
            consequent=("molecular_flexibility", "High"),
            confidence=0.65,
            category="descriptor"
        ),
        Rule(
            name="heavy_atom_threshold",
            description="Heavy Atom Count > 30 indicates complex molecular structure",
            antecedents=[("Heavy_Atom_Count", "?chemical", ">30")],
            consequent=("molecular_complexity", "High"),
            confidence=0.60,
            category="descriptor"
        ),
        # Combined Toxicity Rules
        Rule(
            name="combined_toxic_profile",
            description="logKow>2.5 AND Aromatic_Rings>1 → Increased Toxicity Risk",
            antecedents=[
                ("logKow", "?chemical", ">2.5"),
                ("Aromatic_Rings", "?chemical", ">1")
            ],
            consequent=("toxicity_risk", "High"),
            confidence=0.85,
            category="combined"
        ),
        Rule(
            name="safe_molecule_profile",
            description="MW<300 AND TPSA>60 AND logKow<2 → Safe Profile",
            antecedents=[
                ("MW", "?chemical", "<300"),
                ("TPSA", "?chemical", ">60"),
                ("logKow", "?chemical", "<2")
            ],
            consequent=("toxicity_risk", "Low"),
            confidence=0.80,
            category="combined"
        )
    ]

    def __init__(self, dto_parser=None) -> None:
        """Initialize DTORuleEngine.

        Args:
            dto_parser: DTOParser instance (optional).
        """
        self.dto_parser = dto_parser
        self.rules = self.DESCRIPTOR_RULES.copy()

        logger.info(f"DTORuleEngine initialized with {len(self.rules)} rules")

    def validate_prediction(
        self,
        chemical_id: Optional[str],
        rf_prediction: Dict[str, Any],
        descriptors: Dict[str, float]
    ) -> ValidationResult:
        """Validate RF prediction using Ontology Rules.

        Args:
            chemical_id: Chemical ID.
            rf_prediction: RandomForest prediction result.
            descriptors: Molecular descriptors dictionary.

        Returns:
            ValidationResult object.
        """
        rf_class = rf_prediction.get("class_name", "Unknown")
        rf_confidence = rf_prediction.get("confidence", 0.0)

        # Reduce log verbosity: only log if not Unknown or if confidence is significant
        if rf_class != "Unknown" or rf_confidence > 0.1:
            logger.debug(f"Validating prediction for {chemical_id}: {rf_class} ({rf_confidence:.2%})")

        # Match Rules
        matched_rules = self._match_rules(chemical_id, descriptors)

        # Determine Rule-based Toxicity
        rule_toxicity = self._determine_rule_toxicity(matched_rules)

        # Check Agreement
        agreement = self._check_agreement(rf_class, rule_toxicity)

        # Calculate Confidence
        rule_confidence = self._calculate_rule_confidence(matched_rules)
        combined_confidence = self._combine_confidences(rf_confidence, rule_confidence, agreement)

        # Generate Interpretation
        interpretation = self._generate_interpretation(
            rf_class, rf_confidence, matched_rules, rule_toxicity, agreement
        )

        return ValidationResult(
            chemical_id=chemical_id,
            rf_prediction=rf_class,
            rf_confidence=rf_confidence,
            matched_rules=matched_rules,
            rule_count=len(matched_rules),
            rule_confidence_score=rule_confidence,
            combined_confidence=combined_confidence,
            agreement=agreement,
            interpretation=interpretation
        )

    def _match_rules(
        self,
        chemical_id: Optional[str],
        descriptors: Dict[str, float]
    ) -> List[RuleMatchResult]:
        """Match rules based on descriptors."""
        results = []

        for rule in self.rules:
            matched, met_conditions = self._evaluate_rule(rule, descriptors)

            if matched:
                result = RuleMatchResult(
                    rule_name=rule.name,
                    matched=True,
                    antecedents_met=met_conditions,
                    conclusion=f"{rule.consequent[0]} = {rule.consequent[1]}",
                    confidence=rule.confidence,
                    relevance_to_rf=self._assess_relevance(rule)
                )
                results.append(result)

        return results

    def _evaluate_rule(
        self,
        rule: Rule,
        descriptors: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Evaluate a single rule."""
        met_conditions = []
        all_met = True

        for predicate, subject, condition in rule.antecedents:
            # Get descriptor value
            if predicate not in descriptors:
                all_met = False
                continue

            value = descriptors[predicate]
            
            # Parse and evaluate condition
            is_met = self._evaluate_condition(value, condition)

            if is_met:
                met_conditions.append(f"{predicate}({value}) {condition}")
            else:
                all_met = False

        return all_met, met_conditions

    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate a condition string."""
        try:
            if condition.startswith(">"):
                threshold = float(condition[1:])
                return value > threshold
            elif condition.startswith("<"):
                threshold = float(condition[1:])
                return value < threshold
            elif condition.startswith(">="):
                threshold = float(condition[2:])
                return value >= threshold
            elif condition.startswith("<="):
                threshold = float(condition[2:])
                return value <= threshold
            elif condition.startswith("=="):
                threshold = float(condition[2:])
                return value == threshold
            else:
                return False
        except (ValueError, TypeError):
            return False

    def _determine_rule_toxicity(self, matched_rules: List[RuleMatchResult]) -> str:
        """Determine toxicity level from matched rules."""
        if not matched_rules:
            return "Unknown"

        # Check toxicity rules
        toxic_rules = [r for r in matched_rules if "toxic" in r.rule_name.lower() or "High" in r.conclusion]
        safe_rules = [r for r in matched_rules if "safe" in r.rule_name.lower() or "Low" in r.conclusion]

        if len(toxic_rules) > len(safe_rules):
            return "Toxic"
        elif len(safe_rules) > len(toxic_rules):
            return "Safe"
        else:
            return "Moderate"

    def _check_agreement(self, rf_class: str, rule_toxicity: str) -> str:
        """Check agreement between RF and Rule outcome."""
        if rule_toxicity == "Unknown":
            return "INSUFFICIENT_DATA"

        rf_toxic = rf_class.lower() in ["toxic", "high"]
        rule_toxic = rule_toxicity.lower() in ["toxic", "high"]

        if rf_toxic == rule_toxic:
            return "AGREE"
        else:
            return "DISAGREE"

    def _calculate_rule_confidence(self, matched_rules: List[RuleMatchResult]) -> float:
        """Calculate rule-based confidence score."""
        if not matched_rules:
            return 0.0

        return sum(r.confidence for r in matched_rules) / len(matched_rules)

    def _combine_confidences(
        self,
        rf_confidence: float,
        rule_confidence: float,
        agreement: str
    ) -> float:
        """Combine RF and Rule confidences."""
        if agreement == "AGREE":
            # Increase confidence if agreed
            return min(1.0, (rf_confidence + rule_confidence) / 2 + 0.1)
        elif agreement == "DISAGREE":
            # Decrease confidence if disagreed
            return max(0.0, (rf_confidence + rule_confidence) / 2 - 0.15)
        else:
            # Data insufficient
            return rf_confidence

    def _assess_relevance(self, rule: Rule) -> str:
        """Assess relevance to RF prediction."""
        if "toxicity" in rule.consequent[0].lower():
            return "Directly related to toxicity prediction"
        elif "lipinski" in rule.name.lower():
            return "Drug-likeness Rule - Indirect relation"
        else:
            return "Molecular Property related"

    def _generate_interpretation(
        self,
        rf_class: str,
        rf_confidence: float,
        matched_rules: List[RuleMatchResult],
        rule_toxicity: str,
        agreement: str
    ) -> str:
        """Generate Natural Language Interpretation."""
        lines = []

        lines.append("=" * 50)
        lines.append("       Ontology Rule Validation Report")
        lines.append("=" * 50)
        lines.append("")

        # RF Prediction
        lines.append(f"RandomForest Prediction: {rf_class} ({rf_confidence:.0%})")
        lines.append("")

        # Matched Rules
        if matched_rules:
            lines.append(f"Matched Ontology Rules: {len(matched_rules)}")
            for i, rule in enumerate(matched_rules[:5], 1):
                lines.append(f"  {i}. {rule.rule_name}")
                lines.append(f"      → {rule.conclusion} (Confidence: {rule.confidence:.0%})")
                if rule.antecedents_met:
                    lines.append(f"      Conditions: {'; '.join(rule.antecedents_met[:2])}")
        else:
            lines.append("No Rules Matched")

        lines.append("")

        # Rule-based Decision
        lines.append(f"Rule-based Decision: {rule_toxicity}")
        lines.append("")

        # Agreement
        if agreement == "AGREE":
            lines.append("Result: RF Prediction and Ontology Rules AGREE")
            lines.append("   → High Confidence")
        elif agreement == "DISAGREE":
            lines.append("Result: RF Prediction and Ontology Rules DISAGREE")
            lines.append("   → Further review required (Potential Misclassification)")
        else:
            lines.append("Result: Insufficient Ontology Data")
            lines.append("   → Relying on RF Prediction only")

        lines.append("=" * 50)

        return "\n".join(lines)

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """  ."""
        return [rule.to_dict() for rule in self.rules]


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # 
    engine = DTORuleEngine()

    #   
    rf_prediction = {
        "class_name": "Toxic",
        "confidence": 0.82
    }

    #  (Aspirin )
    descriptors = {
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

    result = engine.validate_prediction("aspirin_001", rf_prediction, descriptors)

    print(result.interpretation)
    print(f"\nAgreement: {result.agreement}")
    print(f"Combined Confidence: {result.combined_confidence:.2%}")

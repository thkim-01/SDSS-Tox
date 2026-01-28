"""
Combined Predictor - ML + Ontology   
- ML    
-     
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """   """
    rule_id: str
    name: str
    category: str
    triggered: bool
    weight: float
    toxicity_direction: str
    interpretation: str
    descriptor_value: float
    threshold_value: float
    detailed_reason: str = ""  #   (: "LogP 4.2 > 3.0 →  ")
    
    
@dataclass
class CombinedPrediction:
    """  """
    smiles: str
    ml_prediction: float
    ontology_score: float
    combined_score: float
    confidence: float
    confidence_level: str
    confidence_action: str
    agreement: str
    triggered_rules: List[RuleResult]
    explanation: str
    shap_features: Optional[List[Dict[str, Any]]] = None


class CombinedPredictor:
    """ML + Ontology  """
    
    #   
    ML_WEIGHT = 0.60
    ONTOLOGY_WEIGHT = 0.40
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: YAML    
        """
        self.config_path = config_path or self._find_config_path()
        self.config = self._load_config()
        self.rules = self._parse_rules()
        self.confidence_levels = self.config.get('confidence_levels', {})
        self.colors = self.config.get('colors', {})
        
        logger.info(f"CombinedPredictor initialized with {len(self.rules)} rules")
        
    def _find_config_path(self) -> str:
        """   """
        possible_paths = [
            "config/ontology_rules.yaml",
            "../config/ontology_rules.yaml",
            "../../config/ontology_rules.yaml",
            os.path.join(os.path.dirname(__file__), "../../../config/ontology_rules.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        #  
        return "config/ontology_rules.yaml"
        
    def _load_config(self) -> Dict:
        """YAML   """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """  """
        return {
            'rules': [
                {
                    'id': 'HIGH_LOGP',
                    'category': 'structural',
                    'name': ' ',
                    'condition': {'type': 'threshold', 'descriptor': 'logp', 'operator': '>', 'value': 3.0},
                    'weight': 0.30,
                    'toxicity_direction': 'positive',
                    'interpretation': 'LogP > 3.0:     '
                },
                {
                    'id': 'HIGH_MW',
                    'category': 'structural',
                    'name': ' ',
                    'condition': {'type': 'threshold', 'descriptor': 'molecular_weight', 'operator': '>', 'value': 500},
                    'weight': 0.15,
                    'toxicity_direction': 'positive',
                    'interpretation': 'MW > 500:  '
                }
            ],
            'confidence_levels': {
                'high': {'range': [0.67, 1.0], 'label': '', 'action': '  '},
                'medium': {'range': [0.34, 0.66], 'label': '', 'action': '  '},
                'low': {'range': [0.0, 0.33], 'label': '', 'action': '  '}
            },
            'colors': {
                'positive': '#FF6B6B',
                'negative': '#4ECDC4',
                'neutral': '#95A5A6'
            }
        }
        
    def _parse_rules(self) -> List[Dict]:
        """ """
        return self.config.get('rules', [])
        
    def evaluate_rules(self, descriptors: Dict[str, float]) -> Tuple[float, List[RuleResult]]:
        """
          
        
        Args:
            descriptors:   
            
        Returns:
            (ontology_score, triggered_rules)
        """
        triggered_rules = []
        total_weight = 0.0
        triggered_weight = 0.0
        
        #    (    )
        desc_map = {
            'molecular_weight': descriptors.get('MW', descriptors.get('molecular_weight', descriptors.get('MolecularWeight', 0))),
            'logp': descriptors.get('logP', descriptors.get('logp', descriptors.get('LogP', 0))),
            'tpsa': descriptors.get('TPSA', descriptors.get('tpsa', descriptors.get('TopoPSA', 0))),
            'num_h_donors': descriptors.get('HBD', descriptors.get('num_h_donors', descriptors.get('NumHDonors', 0))),
            'num_h_acceptors': descriptors.get('HBA', descriptors.get('num_h_acceptors', descriptors.get('NumHAcceptors', 0))),
            'num_rotatable_bonds': descriptors.get('nRotB', descriptors.get('num_rotatable_bonds', descriptors.get('NumRotatableBonds', 0))),
            'num_aromatic_rings': descriptors.get('Aromatic_Rings', descriptors.get('num_aromatic_rings', descriptors.get('NumAromaticRings', 0))),
            'num_heavy_atoms': descriptors.get('Heavy_Atom_Count', descriptors.get('num_heavy_atoms', descriptors.get('NumHeavyAtoms', 0))),
            'num_heteroatoms': descriptors.get('Heteroatom_Count', descriptors.get('num_heteroatoms', descriptors.get('NumHeteroatoms', 0))),
            'fraction_csp3': descriptors.get('fraction_csp3', descriptors.get('FractionCSP3', 0)),
            'molar_refractivity': descriptors.get('molar_refractivity', descriptors.get('MolarRefractivity', 0)),
        }
        
        for rule in self.rules:
            condition = rule.get('condition', {})
            weight = rule.get('weight', 0.1)
            total_weight += weight
            
            #  
            if condition.get('type') == 'threshold':
                descriptor_name = condition.get('descriptor', '')
                operator = condition.get('operator', '>')
                threshold = condition.get('value', 0)
                
                descriptor_value = desc_map.get(descriptor_name, 0)
                triggered = self._evaluate_condition(descriptor_value, operator, threshold)
                
                #   
                detailed_reason = f"{descriptor_name.upper()}: {descriptor_value:.2f} {operator} {threshold} → {rule.get('interpretation', '')}"
                
                rule_result = RuleResult(
                    rule_id=rule.get('id', ''),
                    name=rule.get('name', ''),
                    category=rule.get('category', ''),
                    triggered=triggered,
                    weight=weight,
                    toxicity_direction=rule.get('toxicity_direction', 'positive'),
                    interpretation=rule.get('interpretation', ''),
                    descriptor_value=descriptor_value,
                    threshold_value=threshold,
                    detailed_reason=detailed_reason
                )
                
                if triggered:
                    triggered_weight += weight
                    triggered_rules.append(rule_result)
                    
            elif condition.get('type') == 'composite':
                # Lipinski   
                violations = 0
                sub_conditions = condition.get('sub_conditions', [])
                
                for sub in sub_conditions:
                    sub_desc = desc_map.get(sub.get('descriptor', ''), 0)
                    sub_op = sub.get('operator', '>')
                    sub_val = sub.get('value', 0)
                    
                    if self._evaluate_condition(sub_desc, sub_op, sub_val):
                        violations += 1
                        
                triggered = violations >= 2
                
                # Lipinski   
                detailed_reason = f"Lipinski  : {violations} (≥2  ) → {rule.get('interpretation', '')}"
                
                rule_result = RuleResult(
                    rule_id=rule.get('id', ''),
                    name=rule.get('name', ''),
                    category=rule.get('category', ''),
                    triggered=triggered,
                    weight=weight,
                    toxicity_direction=rule.get('toxicity_direction', 'positive'),
                    interpretation=rule.get('interpretation', ''),
                    descriptor_value=violations,
                    threshold_value=2,
                    detailed_reason=detailed_reason
                )
                
                if triggered:
                    triggered_weight += weight
                    triggered_rules.append(rule_result)
                    
        #    (0-1 )
        if total_weight > 0:
            ontology_score = triggered_weight / total_weight
        else:
            ontology_score = 0.0
            
        return ontology_score, triggered_rules
        
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """ """
        if operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        else:
            return False
            
    def calculate_confidence(self, ml_pred: float, ontology_score: float) -> Tuple[float, str, str]:
        """
         
        
        ML     
        
        Args:
            ml_pred: ML  (0-1)
            ontology_score:   (0-1)
            
        Returns:
            (confidence, level_label, action)
        """
        #    
        difference = abs(ml_pred - ontology_score)
        
        #    (   )
        confidence = 1.0 - difference
        
        #   
        if confidence >= 0.67:
            level = 'high'
        elif confidence >= 0.34:
            level = 'medium'
        else:
            level = 'low'
            
        level_config = self.confidence_levels.get(level, {})
        label = level_config.get('label', level)
        action = level_config.get('action', '')
        
        return confidence, label, action
        
    def predict(
        self, 
        smiles: str,
        descriptors: Dict[str, float],
        ml_prediction: float,
        shap_features: Optional[List[Dict[str, Any]]] = None
    ) -> CombinedPrediction:
        """
          
        
        Args:
            smiles: SMILES 
            descriptors:  
            ml_prediction: ML   (0-1)
            shap_features: SHAP feature  ()
            
        Returns:
            CombinedPrediction 
        """
        # 1.   
        ontology_score, triggered_rules = self.evaluate_rules(descriptors)
        
        # 2.    ( )
        combined_score = (
            self.ML_WEIGHT * ml_prediction + 
            self.ONTOLOGY_WEIGHT * ontology_score
        )
        
        # 3.  
        confidence, confidence_level, confidence_action = self.calculate_confidence(
            ml_prediction, ontology_score
        )
        
        # 4.  
        if abs(ml_prediction - ontology_score) < 0.2:
            agreement = "AGREE"
        elif abs(ml_prediction - ontology_score) < 0.4:
            agreement = "PARTIAL"
        else:
            agreement = "DISAGREE"
            
        # 5.  
        explanation = self._generate_explanation(
            ml_prediction, ontology_score, combined_score,
            confidence, agreement, triggered_rules
        )
        
        return CombinedPrediction(
            smiles=smiles,
            ml_prediction=ml_prediction,
            ontology_score=ontology_score,
            combined_score=combined_score,
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_action=confidence_action,
            agreement=agreement,
            triggered_rules=triggered_rules,
            explanation=explanation,
            shap_features=shap_features
        )
        
    def _generate_explanation(
        self,
        ml_pred: float,
        onto_score: float,
        combined: float,
        confidence: float,
        agreement: str,
        rules: List[RuleResult]
    ) -> str:
        """  """
        
        #   
        if combined >= 0.7:
            risk = "  "
        elif combined >= 0.4:
            risk = "  "
        else:
            risk = "  "
            
        lines = [
            f"[  ]",
            f" : {combined:.2f} ({risk})",
            f"",
            f"[ ]",
            f"- ML : {ml_pred:.2f}",
            f"-  : {onto_score:.2f}",
            f"- : {agreement}",
            f"- : {confidence:.0%}",
            f"",
        ]
        
        if rules:
            lines.append(f"[  ({len(rules)})]")
            for rule in rules[:5]:  #  5
                direction = "Incr" if rule.toxicity_direction == 'positive' else "Decr"
                lines.append(f"- {rule.name} [{direction}]: {rule.interpretation[:50]}...")
                
        if agreement == "DISAGREE":
            lines.append(f"")
            lines.append(f"ML   .   .")
            
        return "\n".join(lines)
        
    def get_rules_summary(self) -> List[Dict]:
        """  """
        return [
            {
                'id': r.get('id'),
                'name': r.get('name'),
                'category': r.get('category'),
                'weight': r.get('weight'),
                'interpretation': r.get('interpretation', '')[:100]
            }
            for r in self.rules
        ]
        
    def get_colors(self) -> Dict[str, str]:
        """  """
        return self.colors
        
        
#  
_combined_predictor = None

def get_combined_predictor() -> CombinedPredictor:
    """Combined Predictor   """
    global _combined_predictor
    if _combined_predictor is None:
        _combined_predictor = CombinedPredictor()
    return _combined_predictor

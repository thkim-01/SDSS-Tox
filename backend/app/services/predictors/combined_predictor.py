"""
Combined Predictor - ML + Ontology 결합 예측 엔진
- ML 예측과 온톨로지 규칙을 결합
- 신뢰도 계산 및 설명 생성
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """단일 규칙 평가 결과"""
    rule_id: str
    name: str
    category: str
    triggered: bool
    weight: float
    toxicity_direction: str
    interpretation: str
    descriptor_value: float
    threshold_value: float
    detailed_reason: str = ""  # 상세 근거 (예: "LogP 4.2 > 3.0 → 높은 지질친화성")
    
    
@dataclass
class CombinedPrediction:
    """결합 예측 결과"""
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
    """ML + Ontology 결합 예측기"""
    
    # 기본 가중치 설정
    ML_WEIGHT = 0.60
    ONTOLOGY_WEIGHT = 0.40
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: YAML 규칙 설정 파일 경로
        """
        self.config_path = config_path or self._find_config_path()
        self.config = self._load_config()
        self.rules = self._parse_rules()
        self.confidence_levels = self.config.get('confidence_levels', {})
        self.colors = self.config.get('colors', {})
        
        logger.info(f"CombinedPredictor initialized with {len(self.rules)} rules")
        
    def _find_config_path(self) -> str:
        """설정 파일 경로 찾기"""
        possible_paths = [
            "config/ontology_rules.yaml",
            "../config/ontology_rules.yaml",
            "../../config/ontology_rules.yaml",
            os.path.join(os.path.dirname(__file__), "../../../config/ontology_rules.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # 기본 경로
        return "config/ontology_rules.yaml"
        
    def _load_config(self) -> Dict:
        """YAML 설정 파일 로드"""
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
        """기본 설정 반환"""
        return {
            'rules': [
                {
                    'id': 'HIGH_LOGP',
                    'category': 'structural',
                    'name': '높은 지질친화도',
                    'condition': {'type': 'threshold', 'descriptor': 'logp', 'operator': '>', 'value': 3.0},
                    'weight': 0.30,
                    'toxicity_direction': 'positive',
                    'interpretation': 'LogP > 3.0: 높은 지질친화성으로 독성 위험 증가'
                },
                {
                    'id': 'HIGH_MW',
                    'category': 'structural',
                    'name': '높은 분자량',
                    'condition': {'type': 'threshold', 'descriptor': 'molecular_weight', 'operator': '>', 'value': 500},
                    'weight': 0.15,
                    'toxicity_direction': 'positive',
                    'interpretation': 'MW > 500: 생체이용률 저하'
                }
            ],
            'confidence_levels': {
                'high': {'range': [0.67, 1.0], 'label': '높음', 'action': '예측 신뢰 가능'},
                'medium': {'range': [0.34, 0.66], 'label': '중간', 'action': '추가 검증 권장'},
                'low': {'range': [0.0, 0.33], 'label': '낮음', 'action': '수동 검토 필수'}
            },
            'colors': {
                'positive': '#FF6B6B',
                'negative': '#4ECDC4',
                'neutral': '#95A5A6'
            }
        }
        
    def _parse_rules(self) -> List[Dict]:
        """규칙 파싱"""
        return self.config.get('rules', [])
        
    def evaluate_rules(self, descriptors: Dict[str, float]) -> Tuple[float, List[RuleResult]]:
        """
        온톨로지 규칙 평가
        
        Args:
            descriptors: 분자 디스크립터 딕셔너리
            
        Returns:
            (ontology_score, triggered_rules)
        """
        triggered_rules = []
        total_weight = 0.0
        triggered_weight = 0.0
        
        # 디스크립터 키 매핑 (지원하는 모든 키 형식 처리)
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
            
            # 조건 평가
            if condition.get('type') == 'threshold':
                descriptor_name = condition.get('descriptor', '')
                operator = condition.get('operator', '>')
                threshold = condition.get('value', 0)
                
                descriptor_value = desc_map.get(descriptor_name, 0)
                triggered = self._evaluate_condition(descriptor_value, operator, threshold)
                
                # 상세 근거 생성
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
                # Lipinski 복합 조건 처리
                violations = 0
                sub_conditions = condition.get('sub_conditions', [])
                
                for sub in sub_conditions:
                    sub_desc = desc_map.get(sub.get('descriptor', ''), 0)
                    sub_op = sub.get('operator', '>')
                    sub_val = sub.get('value', 0)
                    
                    if self._evaluate_condition(sub_desc, sub_op, sub_val):
                        violations += 1
                        
                triggered = violations >= 2
                
                # Lipinski 상세 근거 생성
                detailed_reason = f"Lipinski 위반 수: {violations}개 (≥2개 위반시 트리거) → {rule.get('interpretation', '')}"
                
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
                    
        # 온톨로지 점수 계산 (0-1 범위)
        if total_weight > 0:
            ontology_score = triggered_weight / total_weight
        else:
            ontology_score = 0.0
            
        return ontology_score, triggered_rules
        
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """조건 평가"""
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
        신뢰도 계산
        
        ML과 온톨로지의 일치도에 따라 신뢰도 결정
        
        Args:
            ml_pred: ML 예측값 (0-1)
            ontology_score: 온톨로지 점수 (0-1)
            
        Returns:
            (confidence, level_label, action)
        """
        # 두 예측의 차이 계산
        difference = abs(ml_pred - ontology_score)
        
        # 일치도 기반 신뢰도 (차이가 작을수록 높은 신뢰도)
        confidence = 1.0 - difference
        
        # 신뢰도 레벨 결정
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
        결합 예측 수행
        
        Args:
            smiles: SMILES 문자열
            descriptors: 분자 디스크립터
            ml_prediction: ML 모델 예측값 (0-1)
            shap_features: SHAP feature 기여도 (선택)
            
        Returns:
            CombinedPrediction 결과
        """
        # 1. 온톨로지 규칙 평가
        ontology_score, triggered_rules = self.evaluate_rules(descriptors)
        
        # 2. 결합 점수 계산 (가중 평균)
        combined_score = (
            self.ML_WEIGHT * ml_prediction + 
            self.ONTOLOGY_WEIGHT * ontology_score
        )
        
        # 3. 신뢰도 계산
        confidence, confidence_level, confidence_action = self.calculate_confidence(
            ml_prediction, ontology_score
        )
        
        # 4. 일치도 판단
        if abs(ml_prediction - ontology_score) < 0.2:
            agreement = "AGREE"
        elif abs(ml_prediction - ontology_score) < 0.4:
            agreement = "PARTIAL"
        else:
            agreement = "DISAGREE"
            
        # 5. 설명 생성
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
        """설명 텍스트 생성"""
        
        # 위험 레벨 판단
        if combined >= 0.7:
            risk = "높은 독성 위험"
        elif combined >= 0.4:
            risk = "중간 독성 위험"
        else:
            risk = "낮은 독성 위험"
            
        lines = [
            f"[결합 예측 결과]",
            f"최종 점수: {combined:.2f} ({risk})",
            f"",
            f"[세부 점수]",
            f"- ML 예측: {ml_pred:.2f}",
            f"- 온톨로지 점수: {onto_score:.2f}",
            f"- 일치도: {agreement}",
            f"- 신뢰도: {confidence:.0%}",
            f"",
        ]
        
        if rules:
            lines.append(f"[활성화된 규칙 ({len(rules)}개)]")
            for rule in rules[:5]:  # 상위 5개만
                direction = "Incr" if rule.toxicity_direction == 'positive' else "Decr"
                lines.append(f"- {rule.name} [{direction}]: {rule.interpretation[:50]}...")
                
        if agreement == "DISAGREE":
            lines.append(f"")
            lines.append(f"ML과 온톨로지 의견이 다릅니다. 전문가 검토를 권장합니다.")
            
        return "\n".join(lines)
        
    def get_rules_summary(self) -> List[Dict]:
        """규칙 요약 반환"""
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
        """시각화 색상 반환"""
        return self.colors
        
        
# 싱글톤 인스턴스
_combined_predictor = None

def get_combined_predictor() -> CombinedPredictor:
    """Combined Predictor 싱글톤 인스턴스 반환"""
    global _combined_predictor
    if _combined_predictor is None:
        _combined_predictor = CombinedPredictor()
    return _combined_predictor

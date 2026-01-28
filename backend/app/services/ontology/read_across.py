"""Read-Across 예측 서비스.

Phase 5.1: 4가지 예측 방법 중 하나
- 유사한 구조를 가진 알려진 물질들의 독성 데이터를 기반으로 예측
- 유클리드 거리 또는 코사인 유사도 활용

Author: DTO-DSS Team
Date: 2026-01-19
"""

import logging
import numpy as np
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)


class ReadAcross:
    """Read-Across 예측기.
    
    데이터베이스(CSV)에 있는 유사 물질들을 찾아
    그들의 독성 정보를 바탕으로 타겟 물질의 독성을 예측합니다.
    """

    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: 참조할 데이터셋 경로 (CSV). 없으면 더미 데이터 사용.
        """
        self.database = self._load_database(data_path)
        logger.info(f"ReadAcross initialized with {len(self.database)} reference compounds")

    def _load_database(self, path: str = None) -> pd.DataFrame:
        """데이터베이스 로드 또는 더미 데이터 생성."""
        # 실제 구현에서는 CSV 파일 등에서 로드
        # 여기서는 테스트를 위한 더미 데이터 생성
        
        # 10개 Descriptor + Toxicity Class (0: Safe, 1: Moderate, 2: Toxic)
        dummy_data = [
            # Safe compounds (Aspirin-like)
            {"id": "db_001", "MW": 180.16, "logKow": 1.19, "HBD": 1, "HBA": 4, "nRotB": 3, "TPSA": 63.6, "Aromatic_Rings": 1, "Heteroatom_Count": 4, "Heavy_Atom_Count": 13, "logP": 0.89, "toxicity": 0},
            {"id": "db_002", "MW": 190.2, "logKow": 1.3, "HBD": 1, "HBA": 4, "nRotB": 3, "TPSA": 65.0, "Aromatic_Rings": 1, "Heteroatom_Count": 4, "Heavy_Atom_Count": 14, "logP": 0.95, "toxicity": 0},
            
            # Toxic compounds (High logP, Aromatic)
            {"id": "db_003", "MW": 350.4, "logKow": 4.5, "HBD": 0, "HBA": 2, "nRotB": 5, "TPSA": 30.0, "Aromatic_Rings": 3, "Heteroatom_Count": 2, "Heavy_Atom_Count": 25, "logP": 4.2, "toxicity": 2},
            {"id": "db_004", "MW": 340.1, "logKow": 4.2, "HBD": 0, "HBA": 3, "nRotB": 4, "TPSA": 35.0, "Aromatic_Rings": 3, "Heteroatom_Count": 3, "Heavy_Atom_Count": 24, "logP": 4.0, "toxicity": 2},
            
            # Moderate
            {"id": "db_005", "MW": 250.0, "logKow": 2.5, "HBD": 2, "HBA": 5, "nRotB": 4, "TPSA": 80.0, "Aromatic_Rings": 2, "Heteroatom_Count": 5, "Heavy_Atom_Count": 18, "logP": 2.1, "toxicity": 1},
        ]
        
        return pd.DataFrame(dummy_data)

    def predict(self, descriptors: Dict[str, float], k: int = 3) -> Dict[str, Any]:
        """k-Nearest Neighbors 기반 Read-Across 예측.
        
        Args:
            descriptors: 타겟 물질의 기술자
            k: 참조할 이웃 수
            
        Returns:
            예측 결과
        """
        # 입력 벡터
        target_vector = np.array([
            descriptors.get("MW", 0),
            descriptors.get("logKow", 0),
            descriptors.get("HBD", 0),
            descriptors.get("HBA", 0),
            descriptors.get("nRotB", 0),
            descriptors.get("TPSA", 0),
            descriptors.get("Aromatic_Rings", 0),
            descriptors.get("Heteroatom_Count", 0),
            descriptors.get("Heavy_Atom_Count", 0),
            descriptors.get("logP", 0)
        ])
        
        # DB 벡터
        feature_cols = ["MW", "logKow", "HBD", "HBA", "nRotB", "TPSA", "Aromatic_Rings", "Heteroatom_Count", "Heavy_Atom_Count", "logP"]
        db_vectors = self.database[feature_cols].values
        
        # 유클리드 거리 계산
        distances = np.linalg.norm(db_vectors - target_vector, axis=1)
        
        # 가장 가까운 k개 찾기
        nearest_indices = distances.argsort()[:k]
        nearest_neighbors = self.database.iloc[nearest_indices]
        nearest_distances = distances[nearest_indices]
        
        # 유사도 점수 (거리가 0이면 1, 멀수록 0에 수렴)
        similarities = 1 / (1 + nearest_distances)
        
        # 가중 평균 독성 점수 (0:Safe, 1:Moderate, 2:Toxic) -> 0.0~1.0 스케일로 변환
        # Toxicity 0->0.0, 1->0.5, 2->1.0
        neighbor_scores = nearest_neighbors["toxicity"].values * 0.5
        
        weighted_score = np.average(neighbor_scores, weights=similarities)
        
        # 신뢰도: 가장 가까운 이웃과의 유사도
        confidence = float(similarities[0])
        
        # 상세 정보
        neighbors_info = []
        for i in range(len(nearest_neighbors)):
            neighbors_info.append({
                "id": nearest_neighbors.iloc[i]["id"],
                "toxicity_class": int(nearest_neighbors.iloc[i]["toxicity"]),
                "similarity": round(float(similarities[i]), 3)
            })
            
        prediction_class = "Toxic" if weighted_score > 0.6 else ("Safe" if weighted_score < 0.4 else "Moderate")

        return {
            "method": "Read-Across",
            "score": round(weighted_score, 3),
            "confidence": round(confidence, 2),
            "class": prediction_class,
            "details": {
                "nearest_neighbors": neighbors_info,
                "neighbor_count": k
            }
        }

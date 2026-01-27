from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseModel(ABC):
    """
    모든 예측 모델이 상속받아야 하는 기본 추상 클래스
    """
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """모델을 파일에서 로드"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        예측 수행
        Args:
            features: 입력 특성 (numpy array)
        Returns:
            Dict containing:
            - toxicity_probability: 독성 확률 (0-1)
            - prediction: 예측 클래스 (0 or 1)
            - confidence: 신뢰도 (optional)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """모델 메타데이터 반환 (이름, 버전, 타입 등)"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """모델 타입 식별자 (예: 'rf', 'pytorch', 'svm')"""
        pass

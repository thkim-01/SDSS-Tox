import logging
from typing import Dict, Any, Optional, Type
from .base_model import BaseModel
from .rf_predictor import RFPredictor
from .pytorch_predictor import PyTorchPredictor
from .sdt_predictor import SDTPredictor

logger = logging.getLogger(__name__)

class ModelManager:
    """
    ML 모델 관리자
    - 다양한 유형의 모델 등록 및 관리
    - 런타임에 모델 교체 지원
    """
    
    def __init__(self):
        self._model_types: Dict[str, Type[BaseModel]] = {}
        self._active_model: Optional[BaseModel] = None
        self._models: Dict[str, BaseModel] = {}
        
        # 기본 모델 유형 등록
        self.register_model_type("random_forest", RFPredictor)
        self.register_model_type("pytorch", PyTorchPredictor)
        self.register_model_type("sdt", SDTPredictor)
        
    def register_model_type(self, type_name: str, model_class: Type[BaseModel]):
        """새로운 모델 유형 등록"""
        self._model_types[type_name] = model_class
        logger.info(f"Registered model type: {type_name}")
        
    def load_model(self, model_type: str, model_path: str, set_active: bool = True) -> bool:
        """
        모델 로드 및 (선택적으로) 활성화
        """
        if model_type not in self._model_types:
            logger.error(f"Unknown model type: {model_type}")
            return False
            
        try:
            model_class = self._model_types[model_type]
            model_instance = model_class(model_path)
            
            # 모델 로드 확인 (이미 init에서 로드 시도함)
            if model_instance.get_info()['status'] != 'loaded':
                if not model_instance.load(model_path):
                    return False
            
            self._models[model_type] = model_instance
            
            if set_active:
                self._active_model = model_instance
                logger.info(f"Active model switched to: {model_type}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            return False
            
    def get_active_model(self) -> Optional[BaseModel]:
        """현재 활성 모델 반환"""
        return self._active_model
        
    def get_model_info(self) -> Dict[str, Any]:
        """현재 활성 모델 정보 반환"""
        if not self._active_model:
            return {"status": "no_active_model"}
        return self._active_model.get_info()
        
    def get_supported_types(self) -> list:
        """지원되는 모델 유형 목록 반환"""
        return list(self._model_types.keys())

# 싱글톤 인스턴스
_model_manager = None

def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

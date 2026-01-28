import logging
from typing import Dict, Any, Optional, Type
from .base_model import BaseModel
from .predictors.rf_predictor import RFPredictor
from .predictors.pytorch_predictor import PyTorchPredictor
from .predictors.sdt_predictor import SDTPredictor

logger = logging.getLogger(__name__)

class ModelManager:
    """
    ML  
    -      
    -    
    """
    
    def __init__(self):
        self._model_types: Dict[str, Type[BaseModel]] = {}
        self._active_model: Optional[BaseModel] = None
        self._models: Dict[str, BaseModel] = {}
        
        #    
        self.register_model_type("random_forest", RFPredictor)
        self.register_model_type("pytorch", PyTorchPredictor)
        self.register_model_type("sdt", SDTPredictor)
        
    def register_model_type(self, type_name: str, model_class: Type[BaseModel]):
        """   """
        self._model_types[type_name] = model_class
        logger.info(f"Registered model type: {type_name}")
        
    def load_model(self, model_type: str, model_path: str, set_active: bool = True) -> bool:
        """
           () 
        """
        if model_type not in self._model_types:
            logger.error(f"Unknown model type: {model_type}")
            return False
            
        try:
            model_class = self._model_types[model_type]
            model_instance = model_class(model_path)
            
            #    ( init  )
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
        """   """
        return self._active_model
        
    def get_model_info(self) -> Dict[str, Any]:
        """    """
        if not self._active_model:
            return {"status": "no_active_model"}
        return self._active_model.get_info()
        
    def get_supported_types(self) -> list:
        """    """
        return list(self._model_types.keys())

#  
_model_manager = None

def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

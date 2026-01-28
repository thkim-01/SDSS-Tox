from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseModel(ABC):
    """
           
    """
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """  """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
         
        Args:
            features:   (numpy array)
        Returns:
            Dict containing:
            - toxicity_probability:   (0-1)
            - prediction:   (0 or 1)
            - confidence:  (optional)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """   (, ,  )"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """   (: 'rf', 'pytorch', 'svm')"""
        pass

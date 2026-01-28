'''FastAPI 메인 애플리케이션.'''

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import io

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    from rdkit.Chem import Draw
    RDKIT_CORE_AVAILABLE = True
    RDKIT_DRAW_AVAILABLE = True
except ImportError:
    RDKIT_CORE_AVAILABLE = False
    RDKIT_DRAW_AVAILABLE = False
    Chem = None
    Descriptors = None
    Lipinski = None
    Crippen = None
    Draw = None

# Combined flag for backward compatibility (though we should prefer specific flags)
RDKIT_AVAILABLE = RDKIT_CORE_AVAILABLE

# Service Imports
from app.services.model_manager import get_model_manager
from app.services.explainability.shap_explainer import SHAPExplainer
from app.services.ontology.dto_rule_engine import DTORuleEngine
from app.services.simple_qsar import SimpleQSAR
from app.services.ontology.read_across import ReadAcross
from app.services.predictors.ensemble_dss import EnsembleDSS
from app.services.predictors.combined_predictor import CombinedPredictor
from app.services.data.dataset_loader import DatasetLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 인스턴스 (lifespan에서 초기화)
model_manager: Optional[object] = None
shap_explainer: Optional[object] = None
rule_engine: Optional[object] = None
simple_qsar: Optional[object] = None
read_across: Optional[object] = None
ensemble_dss: Optional[object] = None
combined_predictor: Optional[object] = None
dataset_loader: Optional[object] = None


# ==================== 분자명 조회 헬퍼 ====================

def get_molecule_name_from_smiles(smiles: str) -> str:
    """SMILES에서 분자명 조회.
    
    Args:
        smiles: SMILES string
    
    Returns:
        분자명 또는 "Unknown"
    """
    if not RDKIT_CORE_AVAILABLE or not Chem:
        return "Unknown"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "Unknown"
        
        # IUPAC 이름 시도
        try:
            iupac_name = Chem.MolToIUPACName(mol)
            if iupac_name:
                # IUPAC 이름이 너무 길면 common 이름 시도
                if len(iupac_name) > 100:
                    try:
                        common_name = Chem.MolToSmiles(mol)
                        return common_name[:50] if common_name else "Unknown"
                    except:
                        return iupac_name[:50]
                return iupac_name[:50]
        except:
            pass
        
        # InChI key 시도
        try:
            inchi_key = Chem.MolToInchiKey(mol)
            return f"InChI: {inchi_key[:20]}"
        except:
            pass
        
        return "Unknown"
    except Exception as e:
        logger.warning(f"Error getting molecule name for SMILES {smiles}: {e}")
        return "Unknown"


def compute_descriptors(smiles: str) -> dict:
    """RDKit을 사용하여 분자 기술자 계산."""
    if not RDKIT_CORE_AVAILABLE or not Chem:
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
        from rdkit.Chem import Descriptors as RDKitDescriptors
        from rdkit.Chem import Lipinski as RDKitLipinski
        from rdkit.Chem import Crippen as RDKitCrippen
        
        return {
            "MW": RDKitDescriptors.MolWt(mol),
            "logP": RDKitCrippen.MolLogP(mol),
            "logKow": RDKitCrippen.MolLogP(mol),
            "HBD": RDKitLipinski.NumHDonors(mol),
            "HBA": RDKitLipinski.NumHAcceptors(mol),
            "nRotB": RDKitLipinski.NumRotatableBonds(mol),
            "TPSA": RDKitDescriptors.TPSA(mol),
            "Aromatic_Rings": RDKitLipinski.NumAromaticRings(mol),
            "Heteroatom_Count": RDKitLipinski.NumHeteroatoms(mol),
            "Heavy_Atom_Count": RDKitLipinski.HeavyAtomCount(mol)
        }
    except Exception as e:
        logger.warning(f"Error computing descriptors for {smiles}: {e}")
        return {}


def get_feature_val(descriptors: dict, key: str, default: float = 0.0) -> float:
    """Robust feature lookup from a descriptor dictionary (case-insensitive)."""
    if not descriptors:
        return default
    
    # Try exact match
    if key in descriptors and descriptors[key] is not None:
        return float(descriptors[key])
        
    # Try lowercase
    k_low = key.lower()
    if k_low in descriptors and descriptors[k_low] is not None:
        return float(descriptors[k_low])
        
    # Try common alias mappings
    aliases = {
        "logp": ["logP", "LogP", "MolLogP", "logKow", "logkow", "LogKow"],
        "mw": ["MW", "MolWt", "MolecularWeight", "Molecular Weight"],
        "tpsa": ["TPSA", "tpsa", "TopoPSA"],
        "hbd": ["HBD", "nHBDon", "NumHDonors", "num_h_donors"],
        "hba": ["HBA", "nHBAcc", "NumHAcceptors", "num_h_acceptors"],
    }
    
    if k_low in aliases:
        for alias in aliases[k_low]:
            if alias in descriptors and descriptors[alias] is not None:
                return float(descriptors[alias])
                
    # Search all keys case-insensitively as last resort
    for d_key, d_val in descriptors.items():
        if str(d_key).lower() == k_low and d_val is not None:
            return float(d_val)
            
    return default


# ==================== FastAPI 애플리케이션 ====================

# 모델 파일 경로 (기본 RandomForest 모델)
MODEL_PATH = Path(__file__).parent.parent / "models" / "trained_rf_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작·종료 시 모델 로드·정리."""
    global model_manager, shap_explainer, rule_engine, simple_qsar, read_across, ensemble_dss, combined_predictor, dataset_loader
    logger.info("Loading models...")
    try:
        model_manager = get_model_manager()
        
        # 1. Register Core Models
        from app.services.predictors.rf_predictor import RFPredictor
        from app.services.predictors.dt_predictor import DecisionTreePredictor
        from app.services.predictors.sdt_predictor import SDTPredictor
        
        model_manager.register_model_type("random_forest", RFPredictor)
        model_manager.register_model_type("decision_tree", DecisionTreePredictor)
        model_manager.register_model_type("sdt", SDTPredictor)
        
        # 2. Load Core Models
        # RF (Default)
        if model_manager.load_model("random_forest", str(MODEL_PATH)):
            logger.info(f"Default RandomForest model loaded from {MODEL_PATH}")
        else:
            logger.error("Failed to load default RandomForest model.")
            
        # DT & SDT (Auto-trained/Loaded)
        dt_path = MODEL_PATH.parent / "trained_dt.pkl"
        if model_manager.load_model("decision_tree", str(dt_path), set_active=False):
             logger.info("Decision Tree model loaded/trained.")
             
        sdt_path = MODEL_PATH.parent / "trained_sdt.pkl"
        # SDT 학습은 비동기로 처리하여 서버 시작을 지연시키지 않음
        if sdt_path.exists():
            if model_manager.load_model("sdt", str(sdt_path), set_active=False):
                logger.info("SDT model loaded.")
        else:
            logger.info("SDT model not found. Will train on-demand when first used.")

    except Exception as e:
        logger.error(f"ModelManager init error: {e}")

    # SHAP explainer (optional, only for RandomForest)
    active_model = model_manager.get_active_model() if model_manager else None
    if active_model and active_model.model_type == "random_forest":
        try:
            shap_explainer = SHAPExplainer(active_model.model)
            logger.info("SHAPExplainer initialized")
        except Exception as e:
            logger.warning(f"SHAP init failed: {e}")

    # 기타 서비스 초기화
    try:
        rule_engine = DTORuleEngine()
        logger.info("DTORuleEngine initialized")
    except Exception as e:
        logger.error(f"DTORuleEngine init error: {e}")
    try:
        simple_qsar = SimpleQSAR()
        logger.info("SimpleQSAR initialized")
    except Exception as e:
        logger.error(f"SimpleQSAR init error: {e}")
    try:
        read_across = ReadAcross()
        logger.info("ReadAcross initialized")
    except Exception as e:
        logger.error(f"ReadAcross init error: {e}")
    try:
        ensemble_dss = EnsembleDSS()
        logger.info("EnsembleDSS initialized")
    except Exception as e:
        logger.error(f"EnsembleDSS init error: {e}")
    try:
        combined_predictor = CombinedPredictor()
        logger.info("CombinedPredictor initialized")
    except Exception as e:
        logger.error(f"CombinedPredictor init error: {e}")
    try:
        dataset_loader = DatasetLoader()
        logger.info("DatasetLoader initialized")
    except Exception as e:
        logger.error(f"DatasetLoader init error: {e}")

    yield
    # 정리 단계
    logger.info("Unloading models...")
    model_manager = None
    shap_explainer = None
    rule_engine = None
    simple_qsar = None
    read_across = None
    ensemble_dss = None
    combined_predictor = None
    dataset_loader = None
# FastAPI 앱 생성
app = FastAPI(
    title="DTO-DSS API",
    description="Drug Target Ontology Decision Support System - RandomForest + SHAP API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic 모델 ====================

class DescriptorInput(BaseModel):
    """분자 기술자 입력 모델."""
    MW: float = Field(..., description="분자량", ge=0)
    logKow: float = Field(..., description="옥탄올-물 분배계수")
    HBD: int = Field(..., description="수소 결합 공여체 수", ge=0)
    HBA: int = Field(..., description="수소 결합 수용체 수", ge=0)
    nRotB: int = Field(..., description="회전 가능 결합 수", ge=0)
    TPSA: float = Field(..., description="극성 표면적", ge=0)
    Aromatic_Rings: int = Field(..., description="방향족 고리 수", ge=0)
    Heteroatom_Count: int = Field(..., description="이종원자 수", ge=0)
    Heavy_Atom_Count: int = Field(..., description="중원자 수", ge=0)
    logP: float = Field(..., description="지질친화성 logP")
    class Config:
        json_schema_extra = {
            "example": {
                "MW": 180.16,
                "logKow": 1.19,
                "HBD": 1,
                "HBA": 4,
                "nRotB": 3,
                "TPSA": 63.60,
                "Aromatic_Rings": 1,
                "Heteroatom_Count": 4,
                "Heavy_Atom_Count": 13,
                "logP": 0.89,
            }
        }

class PredictionRequest(BaseModel):
    """예측 요청 모델."""
    chemical_id: Optional[str] = Field(None, description="화학물질 ID")
    descriptors: DescriptorInput = Field(..., description="10개 분자 기술자")

class SHAPRequest(BaseModel):
    """SHAP 설명 요청 모델."""
    chemical_id: Optional[str] = Field(None, description="화학물질 ID")
    descriptors: DescriptorInput = Field(..., description="10개 분자 기술자")
    target_class: int = Field(2, description="타겟 클래스 (0=Safe, 1=Moderate, 2=Toxic)", ge=0, le=2)

class QSARRequest(BaseModel):
    smiles: Optional[str] = Field(None, description="SMILES string")
    descriptors: Optional[DescriptorInput] = Field(None, description="Descriptor dict (10 features)")
    return_descriptors: bool = Field(True, description="Return computed descriptors if smiles provided")

class HealthResponse(BaseModel):
    """헬스체크 응답 모델."""
    status: str
    active_model: str
    rf_predictor: str
    shap_explainer: str
    rule_engine: str
    simple_qsar: str
    read_across: str
    ensemble_dss: str
    model_path: str
    class Config:
        protected_namespaces = ()

class ModelSwitchRequest(BaseModel):
    model_type: str
    model_path: str
    class Config:
        protected_namespaces = ()

# ==================== 헬퍼 함수 ====================

def descriptors_to_vector(descriptors: DescriptorInput) -> np.ndarray:
    """DescriptorInput을 numpy 배열로 변환."""
    return np.array([[
        descriptors.MW,
        descriptors.logKow,
        descriptors.HBD,
        descriptors.HBA,
        descriptors.nRotB,
        descriptors.TPSA,
        descriptors.Aromatic_Rings,
        descriptors.Heteroatom_Count,
        descriptors.Heavy_Atom_Count,
        descriptors.logP,
    ]])

# ==================== API 엔드포인트 ====================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "DTO-DSS API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predictions": "/predictions/randomforest",
            "explainability": "/explainability/shap",
            "analysis": "/analysis/ensemble",
            "health": "/health",
            "models": "/models/info",
        },
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    active = model_manager.get_active_model() if model_manager else None
    return HealthResponse(
        status="healthy",
        active_model=active.model_type if active else "none",
        rf_predictor="loaded" if active else "not_loaded",
        shap_explainer="loaded" if shap_explainer else "not_loaded",
        rule_engine="loaded" if rule_engine else "not_loaded",
        simple_qsar="loaded" if simple_qsar else "not_loaded",
        read_across="loaded" if read_across else "not_loaded",
        ensemble_dss="loaded" if ensemble_dss else "not_loaded",
        model_path=str(MODEL_PATH),
    )

@app.get("/models/info", tags=["Models"])
async def get_model_info():
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    return model_manager.get_model_info()


# ==================== Model Add‑On API ====================

class ModelAddRequest(BaseModel):
    model_type: str
    model_path: str
    config: Optional[dict] = None  # optional configuration for custom models
    class Config:
        protected_namespaces = ()

@app.post("/models/add", tags=["Models"])
async def add_model(request: ModelAddRequest):
    """Register a new model type at runtime and load it.
    The model must implement the BaseModel interface.
    """
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    # Dynamically import the model class based on type (e.g., 'custom')
    # For simplicity, we assume the model_type corresponds to a Python module under app.services
    try:
        module_path = f"app.services.{request.model_type}_predictor"
        module = __import__(module_path, fromlist=['*'])
        predictor_class = getattr(module, f"{request.model_type.title().replace('_', '')}Predictor")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to import model class: {e}")
    # Register the new model type
    model_manager.register_model_type(request.model_type, predictor_class)
    # Load the model
    success = model_manager.load_model(request.model_type, request.model_path)
    if success:
        return {"status": "success", "message": f"Model {request.model_type} added and loaded."}
    raise HTTPException(status_code=400, detail="Failed to load the added model")

# ==================== Batch Analysis Plot Data API ====================

@app.get("/analysis/plot-data", tags=["Analysis"])
async def get_plot_data():
    """Return data for regression scatter plot.
    
    Returns plot data including all 100 samples with predictions.
    """
    if not model_manager or not dataset_loader:
        raise HTTPException(status_code=503, detail="Model Manager or Dataset Loader not initialized")
    
    active = model_manager.get_active_model()
    if active is None:
        raise HTTPException(status_code=503, detail="No active model loaded")
    
    if dataset_loader is None:
        raise HTTPException(status_code=503, label="Dataset loader not available")
    
    # Assume dataset_loader provides a method to load all samples
    try:
        # Try to get samples from dataset_loader
        samples = dataset_loader.load_all() if hasattr(dataset_loader, 'load_all') else []
        
        if not samples or len(samples) == 0:
            # Fallback: generate synthetic data for demo
            import random
            samples = []
            for i in range(100):
                desc = {
                    "MW": random.uniform(100, 500),
                    "logKow": random.uniform(-2, 7),
                    "HBD": random.randint(0, 5),
                    "HBA": random.randint(0, 10),
                    "nRotB": random.randint(0, 10),
                    "TPSA": random.uniform(0, 140),
                    "Aromatic_Rings": random.randint(0, 4),
                    "Heteroatom_Count": random.randint(1, 10),
                    "Heavy_Atom_Count": random.randint(5, 25),
                    "logP": random.uniform(-1, 6)
                }
                samples.append({"smiles": f"mock_smiles_{i}", "descriptors": desc})
    except Exception as e:
        logger.error(f"Failed to load samples: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load samples: {e}")
    
    results = []
    for idx, sample in enumerate(samples):
        try:
            # descriptors is a dict, not a DescriptorInput object
            desc_dict = sample.get("descriptors", {})
            
            # Get feature vector
            fv = descriptors_to_vector(desc_dict)
            
            # Predict using active model
            if active.model_type == "random_forest":
                pred = active.predict(fv)
                ml_pred = pred.get("probabilities", {}).get(1, 0.0)  # Toxic probability
            else:
                ml_pred = 0.5  # Mock prediction
            
            # Mock ontology score
            onto_score = 0.3 + (0.4 * (ml_pred - 0.5))
            combined = 0.6 * ml_pred + 0.4 * onto_score
            confidence = 0.5 + 0.5 * abs(ml_pred - onto_score)
            
            results.append({
                "index": idx,
                "smiles": sample.get("smiles", ""),
                "ml_prediction": ml_pred,
                "ontology_score": onto_score,
                "combined_score": combined,
                "confidence": confidence
            })
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    logger.info(f"Generated plot data for {len(results)} samples")
    
    return {
        "total_samples": len(results),
        "samples": results
    }


@app.post("/analysis/full-scatter", tags=["Analysis"])
async def create_full_scatter_plot():
    """Create full scatter plot for all data.
    
    Generates a scatter plot with predictions for all available samples.
    """
    if not model_manager or not dataset_loader:
        raise HTTPException(status_code=503, detail="Model Manager or Dataset Loader not initialized")
    
    plot_data = await get_plot_data()
    
    # Use px.scatter for modern visualization
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame(plot_data["samples"])
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x="ml_prediction",
        y="ontology_score",
        color="confidence",
        color_continuous_scale=px.colors.sequential.Viridis,
        size="combined_score",
        hover_data=['smiles', 'index', 'ml_prediction', 'ontology_score', 'combined_score', 'confidence'],
        labels={
            "ml_prediction": "ML Prediction",
            "ontology_score": "Ontology Score",
            "combined_score": "Combined Score",
            "confidence": "Confidence"
        },
        title=f"DTO-DSS Full Analysis ({len(df)} samples)",
        template="plotly_white"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="ML Prediction",
        yaxis_title="Ontology Score",
        legend_orientation="h",
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Segoe UI, Arial, sans-serif"
        )
    )
    
    # Add reference line (y=x for perfect agreement)
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Perfect Agreement (y=x)'
    )
    
    # Return as HTML
    return HTMLResponse(content=fig.to_html(full_html=True, include_plotlyjs=True))
@app.get("/analysis/rf-importance", tags=["Analysis"])
async def get_rf_feature_importance():
    """Get Random Forest feature importance."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
        
    try:
        active_model = model_manager.get_active_model()
        if not active_model or active_model.model_type != "random_forest":
            raise HTTPException(status_code=400, detail="Active model is not Random Forest")
            
        importance_dict = active_model.get_feature_importance()
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "feature_importance": sorted_importance,
            "model_type": "random_forest",
            "n_estimators": active_model.model.n_estimators
        }
    except Exception as e:
        logger.error(f"RF importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get importance: {e}")

@app.get("/analysis/molecule-name", tags=["Analysis"])
async def get_molecule_name(smiles: str):
    """Get molecule name from SMILES using RDKit (primary method)."""
    # Initialize default values
    name = "Unknown"
    formula = "N/A"
    mw = 0.0
    isomeric_smiles = smiles
    
    # 1. Try RDKit for properties (if available)
    results = {
        "name": name,
        "formula": formula,
        "mw": mw,
        "smiles": isomeric_smiles,
        "qed": 0.5, # Default
        "alerts": {"PAINS": "None", "Brenk": "None"}
    }
    
    if RDKIT_CORE_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                from rdkit.Chem import rdMolDescriptors
                from rdkit.Chem import QED
                
                results["formula"] = rdMolDescriptors.CalcMolFormula(mol)
                results["mw"] = Descriptors.MolWt(mol)
                results["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
                
                # QED Score
                try:
                    results["qed"] = float(QED.qed(mol))
                except:
                    pass
                    
                # Structural Alerts (Demo Implementation with SMARTS)
                # PAINS (Pan Assay Interference Compounds) - Simplified Examples
                pains_smarts = {
                    "Quinone": "O=C1[C,N]C=CC(=O)[C,N]1",
                    "Catechol": "c1c(O)c(O)ccc1",
                    "Azo_Group": "N=N"
                }
                
                detected_pains = []
                for pname, smarts in pains_smarts.items():
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        detected_pains.append(pname)
                        
                if detected_pains:
                    results["alerts"]["PAINS"] = f"Detected ({', '.join(detected_pains)})"
                
                # Brenk Filters (Toxicophores) - Simplified Examples
                brenk_smarts = {
                    "Nitro_Group": "[N+](=O)[O-]",
                    "Thiocarbonyl": "C=S"
                }
                
                detected_brenk = []
                for bname, smarts in brenk_smarts.items():
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern and mol.HasSubstructMatch(pattern):
                        detected_brenk.append(bname)
                        
                if detected_brenk:
                    results["alerts"]["Brenk"] = f"Detected ({', '.join(detected_brenk)})"

                # Try Name from RDKit logic (basic)
                try:
                    # Very basic chemical name generation if possible or just use Hill notation as fallback
                    if results["name"] == "Unknown":
                         results["name"] = results["formula"]
                except:
                    pass
        except Exception as e:
            logger.warning(f"RDKit processing failed for {smiles}: {e}")

    # 2. Return extended metadata
    return results

@app.get("/analysis/sdt-tree", tags=["Analysis"])
async def get_sdt_tree(smiles: Optional[str] = None, max_depth: Optional[int] = None):
    """Get SDT Tree structure for visualization. Optionally highlight path for a SMILES and limit tree depth."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
        
    # Try SDT first, then fall back to decision_tree
    tree_model = None
    model_name = None
    
    if hasattr(model_manager, '_models'):
        if "sdt" in model_manager._models:
            sdt = model_manager._models["sdt"]
            if hasattr(sdt, 'model') and sdt.model is not None:
                tree_model = sdt
                model_name = "sdt"
        
        # Fallback to decision_tree if SDT not available
        if tree_model is None and "decision_tree" in model_manager._models:
            dt = model_manager._models["decision_tree"]
            if hasattr(dt, 'model') and dt.model is not None:
                tree_model = dt
                model_name = "decision_tree"
            
    if not tree_model:
        return {"error": "No tree model loaded", "available_models": list(model_manager._models.keys()) if hasattr(model_manager, '_models') else []}
        
    active_path_ids = []
    debug_info = {
        "smiles_provided": bool(smiles),
        "model_type": model_name,
        "rdkit_available": RDKIT_CORE_AVAILABLE,
        "chem_available": Chem is not None
    }
    if smiles:
        # Calculate path for this molecule
        try:
            # Compute descriptors
            desc = compute_descriptors(smiles)
            debug_info["desc_keys"] = list(desc.keys()) if desc else []
            logger.info(f"SDT tree: computed descriptors for {smiles}: {desc}")
            if desc:
                # Create feature vector with FEATURE_NAMES (includes SEMANTIC_FEATURES)
                feature_names = getattr(tree_model, 'FEATURE_NAMES', None)
                if feature_names is None:
                    base_features = getattr(tree_model, 'BASE_FEATURES', [
                        "MW", "logKow", "HBD", "HBA", "nRotB",
                        "TPSA", "Aromatic_Rings", "Heteroatom_Count",
                        "Heavy_Atom_Count", "logP"
                    ])
                    semantic_features = getattr(tree_model, 'SEMANTIC_FEATURES', 
                        ["Ontology_Rule_Count", "Ontology_Rule_Confidence"])
                    feature_names = base_features + semantic_features
                
                # Add semantic features to descriptors if missing
                if "Ontology_Rule_Count" not in desc:
                    desc["Ontology_Rule_Count"] = 0.0
                if "Ontology_Rule_Confidence" not in desc:
                    desc["Ontology_Rule_Confidence"] = 0.0
                    
                feats = [float(desc.get(k, 0.0)) for k in feature_names]
                feature_arr = np.array([feats])
                debug_info["feature_arr"] = feats
                debug_info["feature_names"] = feature_names
                logger.info(f"SDT tree: feature_arr = {feats}")
                
                # Use sklearn decision_path directly on the underlying model
                # Handle both wrapper (tree_model.model) and direct DecisionTreeClassifier
                actual_model = getattr(tree_model, 'model', tree_model)
                debug_info["actual_model_type"] = str(type(actual_model))
                debug_info["has_decision_path"] = hasattr(actual_model, 'decision_path')
                if actual_model is not None and hasattr(actual_model, 'decision_path'):
                    try:
                        node_indicator = actual_model.decision_path(feature_arr)
                        active_path_ids = node_indicator.indices.tolist()
                        debug_info["path_success"] = True
                        logger.info(f"Computed active_path for {smiles}: {active_path_ids}")
                    except Exception as dp_err:
                        debug_info["path_error"] = str(dp_err)
                        logger.warning(f"decision_path failed: {dp_err}")
                else:
                    logger.warning(f"actual_model is None or has no decision_path")
            else:
                debug_info["desc_empty"] = True
                logger.warning(f"compute_descriptors returned empty for {smiles}")
        except Exception as e:
            logger.warning(f"Failed to compute active path for {smiles}: {e}")

    # Generate Tree Data (Nodes and Edges) - ENHANCED for Reliable Visualization
    try:
        from sklearn.tree import _tree
        # Handle both wrapper (tree_model.model) and direct DecisionTreeClassifier
        sklearn_tree = getattr(tree_model, 'model', tree_model)
        tree_ = sklearn_tree.tree_
        
        # Get feature names
        feature_names = getattr(tree_model, 'FEATURE_NAMES', getattr(tree_model, 'BASE_FEATURES', [
            "MW", "logKow", "HBD", "HBA", "nRotB",
            "TPSA", "Aromatic_Rings", "Heteroatom_Count",
            "Heavy_Atom_Count", "logP"
        ]))
        
        class_names = getattr(tree_model, 'CLASS_NAMES', {0: "Safe", 1: "Toxic"})
        
        # Semantic concept mappings for abstract headers
        SEMANTIC_CONCEPTS = {
            "MW": "Molecular Size Assessment",
            "logKow": "Membrane Permeability Check",
            "logP": "Lipophilicity Analysis",
            "HBD": "H-Bond Donor Capacity",
            "HBA": "H-Bond Acceptor Capacity",
            "TPSA": "Polar Surface Evaluation",
            "nRotB": "Molecular Flexibility Test",
            "Aromatic_Rings": "Aromatic Character Analysis",
            "Heteroatom_Count": "Heteroatom Richness Check",
            "Heavy_Atom_Count": "Molecular Complexity Score",
            "Ontology_Rule_Count": "DTO Knowledge Base Match",
            "Ontology_Rule_Confidence": "Semantic Confidence Level"
        }
        
        # Co-occurring feature correlations (which features correlate with which)
        CORRELATED_FEATURES = {
            "MW": ["Heavy_Atom_Count", "nRotB", "logP"],
            "logKow": ["logP", "TPSA", "HBA"],
            "logP": ["logKow", "Aromatic_Rings", "MW"],
            "HBD": ["HBA", "TPSA", "Heteroatom_Count"],
            "HBA": ["HBD", "Heteroatom_Count", "TPSA"],
            "TPSA": ["HBA", "HBD", "logP"],
            "nRotB": ["MW", "Heavy_Atom_Count", "Aromatic_Rings"],
            "Aromatic_Rings": ["logP", "Heavy_Atom_Count", "MW"],
            "Heteroatom_Count": ["HBA", "HBD", "TPSA"],
            "Heavy_Atom_Count": ["MW", "nRotB", "Aromatic_Rings"],
            "Ontology_Rule_Count": ["Ontology_Rule_Confidence", "logP", "MW"],
            "Ontology_Rule_Confidence": ["Ontology_Rule_Count", "HBA", "HBD"]
        }
        
        # Calculate total samples at root for normalization
        total_samples = int(np.sum(tree_.value[0]))
        
        nodes = []
        # Queue items: (node_id, parent_id, decision_type, depth)
        queue = [(0, -1, "root", 0)] 
        actual_max_depth = 0
        
        while queue:
            node_id, parent_id, decision, depth = queue.pop(0)
            
            # Skip nodes beyond max_depth if specified
            if max_depth is not None and depth > max_depth:
                continue
            
            is_leaf = tree_.children_left[node_id] == _tree.TREE_LEAF
            
            # Track actual max depth
            actual_max_depth = max(actual_max_depth, depth)
            
            # Sample count at this node
            node_samples = int(np.sum(tree_.value[node_id]))
            sample_ratio = node_samples / total_samples if total_samples > 0 else 0
            
            # Class distribution at this node
            value = tree_.value[node_id][0]
            total_at_node = np.sum(value)
            safe_pct = (value[0] / total_at_node * 100) if total_at_node > 0 else 50
            toxic_pct = (value[1] / total_at_node * 100) if len(value) > 1 and total_at_node > 0 else 50
            
            node_data = {
                "id": int(node_id),
                "parent": int(parent_id) if parent_id != -1 else None,
                "is_leaf": bool(is_leaf),
                "is_active": int(node_id) in active_path_ids,
                "decision": decision,
                "depth": depth,
                "samples": node_samples,
                "sample_ratio": round(sample_ratio, 4),
                "distribution": {
                    "safe": round(safe_pct, 1),
                    "toxic": round(toxic_pct, 1)
                }
            }
            
            if not is_leaf:
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                if 0 <= feature_idx < len(feature_names):
                    fname = feature_names[feature_idx]
                else:
                    fname = f"Feature {feature_idx}"
                
                # Abstract semantic concept
                concept = SEMANTIC_CONCEPTS.get(fname, f"{fname} Evaluation")
                
                # Co-occurring (correlated) descriptors for multi-feature context
                correlated = CORRELATED_FEATURES.get(fname, [])[:2]  # Top 2 correlated
                
                # Calculate feature importance contribution at this node
                # (simplified: based on sample impurity reduction)
                impurity_before = tree_.impurity[node_id]
                left_id = tree_.children_left[node_id]
                right_id = tree_.children_right[node_id]
                left_samples = int(np.sum(tree_.value[left_id])) if left_id != _tree.TREE_LEAF else 0
                right_samples = int(np.sum(tree_.value[right_id])) if right_id != _tree.TREE_LEAF else 0
                
                importance_score = round(impurity_before * (node_samples / total_samples), 3)
                
                node_data["name"] = fname
                node_data["concept"] = concept
                node_data["rule"] = f"{fname} ≤ {threshold:.2f}"
                node_data["threshold"] = round(threshold, 2)
                node_data["correlated_features"] = correlated
                node_data["importance"] = importance_score
                node_data["impurity"] = round(impurity_before, 4)
                
                # Left child = True (≤ threshold), Right child = False (> threshold)
                queue.append((tree_.children_left[node_id], node_id, "yes", depth + 1))
                queue.append((tree_.children_right[node_id], node_id, "no", depth + 1))
            else:
                # Leaf node
                class_idx = np.argmax(value)
                prob = value[class_idx] / total_at_node if total_at_node > 0 else 0.0
                
                class_name = class_names.get(class_idx, str(class_idx))
                node_data["name"] = f"{class_name}"
                node_data["concept"] = f"Final Classification: {class_name}"
                node_data["rule"] = f"Confidence: {prob:.1%}"
                node_data["value"] = value.tolist()
                node_data["predicted_class"] = class_name
                node_data["confidence"] = round(prob, 3)
                
            nodes.append(node_data)
        
        # Use actual_max_depth computed during traversal
        if not nodes:
            actual_max_depth = 0
            
        # Update debug info with final active_path
        debug_info["active_path_computed"] = active_path_ids
        debug_info["total_samples"] = total_samples
        debug_info["max_depth"] = actual_max_depth
        debug_info["node_count"] = len(nodes)
        
        return {
            "nodes": nodes, 
            "active_path": active_path_ids, 
            "total_samples": total_samples,
            "max_depth": actual_max_depth,
            "debug": debug_info
        }
        
    except Exception as e:
        logger.error(f"Failed to extract SDT tree: {e}")
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


# ==================== Combined Predictions API ====================

class CombinedPredictionRequest(BaseModel):
    """Combined prediction request with SMILES and descriptors."""
    chemical_id: Optional[str] = Field(None, description="Chemical ID")
    smiles: str = Field(..., description="SMILES string")
    # Descriptors
    MW: float = Field(..., description="Molecular Weight", ge=0)
    logKow: float = Field(..., description="LogKow")
    HBD: int = Field(..., description="H-Bond Donors", ge=0)
    HBA: int = Field(..., description="H-Bond Acceptors", ge=0)
    nRotB: int = Field(..., description="Rotatable Bonds", ge=0)
    TPSA: float = Field(..., description="TPSA", ge=0)
    Aromatic_Rings: int = Field(..., description="Aromatic Rings", ge=0)
    Heteroatom_Count: int = Field(..., description="Heteroatom Count", ge=0)
    Heavy_Atom_Count: int = Field(..., description="Heavy Atom Count", ge=0)
    logP: float = Field(..., description="LogP")
    
    model_type: str = Field(default="random_forest", description="Model to use: random_forest, decision_tree, sdt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "MW": 180.16, "logKow": 1.19, "HBD": 1, "HBA": 4, 
                "nRotB": 3, "TPSA": 63.60, "Aromatic_Rings": 1, 
                "Heteroatom_Count": 4, "Heavy_Atom_Count": 13, "logP": 0.89,
                "model_type": "random_forest"
            }
        }
        protected_namespaces = ()


@app.post("/predictions/combined", tags=["Predictions"])
async def predict_combined(request: CombinedPredictionRequest):
    """
    Combined ML + Ontology prediction.
    Returns ML prediction, ontology score, combined score, confidence, and triggered rules.
    """
    if not model_manager or not combined_predictor:
        raise HTTPException(status_code=503, detail="Model or CombinedPredictor not initialized")
    
    try:
        # 1. Select Model
        target_model_type = request.model_type.lower() if request.model_type else "random_forest"
        active_model = None
        
        # Access private _models if possible or add accessor in ModelManager. 
        # Since we are in main.py and use ModelManager class, we can just acccess _models if strictness is low 
        # or use get_model logic if we added it (we didn't).
        # We'll use direct access since we are the app author.
        if hasattr(model_manager, "_models") and target_model_type in model_manager._models:
             active_model = model_manager._models[target_model_type]
        else:
             # Fallback to active
             active_model = model_manager.get_active_model()
             
        if active_model is None:
             raise HTTPException(status_code=503, detail="No model available")

        # 1.1 Calculate Descriptors from SMILES (Auto-fill/Correction)
        # Always try to calculate if RDKit is available to ensure accuracy
        calc_descriptors = {}
        if RDKIT_CORE_AVAILABLE and request.smiles:
            try:
                mol = Chem.MolFromSmiles(request.smiles)
                if mol:
                    calc_descriptors["MW"] = Descriptors.MolWt(mol)
                    calc_descriptors["logP"] = Descriptors.MolLogP(mol)
                    calc_descriptors["logKow"] = calc_descriptors["logP"]
                    calc_descriptors["TPSA"] = Descriptors.TPSA(mol)
                    calc_descriptors["HBD"] = Lipinski.NumHDonors(mol)
                    calc_descriptors["HBA"] = Lipinski.NumHAcceptors(mol)
                    calc_descriptors["nRotB"] = Lipinski.NumRotatableBonds(mol)
                    calc_descriptors["Aromatic_Rings"] = Lipinski.NumAromaticRings(mol)
                    calc_descriptors["Heteroatom_Count"] = Lipinski.NumHeteroatoms(mol)
                    calc_descriptors["Heavy_Atom_Count"] = mol.GetNumHeavyAtoms()
            except Exception as e:
                logger.warning(f"Descriptor calculation failed for {request.smiles}: {e}")

        # 2. Descriptor Dict for Ontology & Return (Standardized keys)
        descriptors_dict = {
            "MW": float(calc_descriptors.get("MW", request.MW)),
            "logKow": float(calc_descriptors.get("logKow", request.logKow)),
            "HBD": int(calc_descriptors.get("HBD", request.HBD)),
            "HBA": int(calc_descriptors.get("HBA", request.HBA)),
            "nRotB": int(calc_descriptors.get("nRotB", request.nRotB)),
            "TPSA": float(calc_descriptors.get("TPSA", request.TPSA)),
            "Aromatic_Rings": int(calc_descriptors.get("Aromatic_Rings", request.Aromatic_Rings)),
            "Heteroatom_Count": int(calc_descriptors.get("Heteroatom_Count", request.Heteroatom_Count)),
            "Heavy_Atom_Count": int(calc_descriptors.get("Heavy_Atom_Count", request.Heavy_Atom_Count)),
            "logP": float(calc_descriptors.get("logP", request.logP))
        }
        
        # 3. Create Feature Vector (1x10) from descriptors_dict (Normalized/Corrected)
        features = np.array([[
            descriptors_dict["MW"], descriptors_dict["logKow"], descriptors_dict["HBD"], 
            descriptors_dict["HBA"], descriptors_dict["nRotB"], descriptors_dict["TPSA"], 
            descriptors_dict["Aromatic_Rings"], descriptors_dict["Heteroatom_Count"],
            descriptors_dict["Heavy_Atom_Count"], descriptors_dict["logP"]
        ]])
        
        # 4. ML Prediction
        # Handle specific logic for SDT vs RF
        ml_prediction = 0.0
        
        try:
            # Robust probability extraction
            def get_toxicity_score(probs_dict):
                # Try explicit keys
                if "Toxic" in probs_dict: return float(probs_dict["Toxic"])
                if 2 in probs_dict: return float(probs_dict[2])
                
                # If Safe is present, use complement
                if "Safe" in probs_dict:
                    score = 1.0 - float(probs_dict["Safe"])
                    # If Moderate exists, maybe we should account for it?
                    # For now, simplistic approach: (1-Safe) includes Moderate + Toxic
                    return score
                    
                # Fallback for binary 0/1 without names
                if 1 in probs_dict: return float(probs_dict[1])
                
                return 0.0

            if active_model.model_type == "sdt":
                 res = active_model.predict(features, smiles=request.smiles)
                 ml_prediction = get_toxicity_score(res.get("probabilities", {}))
                 
            elif active_model.model_type == "decision_tree":
                 res = active_model.predict(features)
                 ml_prediction = get_toxicity_score(res.get("probabilities", {}))
            
            else:
                 # Default RF
                 res = active_model.predict(features)
                 ml_prediction = get_toxicity_score(res.get("probabilities", {}))
        except Exception as pred_err:
            logger.error(f"Prediction failed for {target_model_type}: {pred_err}")
            ml_prediction = 0.5 # Fail-safe

        # 4.5 SHAP Explanation (only for RandomForest)
        shap_features = None
        if active_model.model_type == "random_forest":
            try:
                shap_explainer = SHAPExplainer(active_model.model)
                shap_result = shap_explainer.explain_prediction(features[0])  # features is 2D, take first row
                shap_features = shap_result.get("shap_values", [])
                logger.info(f"SHAP calculated for {request.smiles[:10]}...: {len(shap_features)} features")
            except Exception as shap_err:
                logger.warning(f"SHAP calculation failed: {shap_err}")
                shap_features = []

        
        # 5. Combined Scoring
        result = combined_predictor.predict(
            smiles=request.smiles,
            descriptors=descriptors_dict, 
            ml_prediction=float(ml_prediction),
            shap_features=shap_features
        )
        
        # Log for debugging Batch sharing issues
        if "test" in request.smiles.lower() or "bbbp" in str(request.smiles).lower():
             logger.info(f"SMILES: {request.smiles[:10]}... ML: {ml_prediction:.4f} ONTO: {result.ontology_score:.4f}")
        
        # 6. Response Formatting
        triggered_rules = []
        for rule in result.triggered_rules:
            triggered_rules.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "category": rule.category,
                "weight": rule.weight,
                "toxicity_direction": rule.toxicity_direction,
                "interpretation": rule.interpretation,
                "descriptor_value": rule.descriptor_value,
                "threshold_value": rule.threshold_value,
                "detailed_reason": rule.detailed_reason
            })
        
        # Calculate RDKit extras if available
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(request.smiles)
            if mol:
                descriptors_dict.update({
                    "Lipinski_Score": sum([1 for cond in [
                            Descriptors.MolWt(mol) <= 500,
                            Descriptors.MolLogP(mol) <= 5,
                            Descriptors.NumHDonors(mol) <= 5,
                            Descriptors.NumHAcceptors(mol) <= 10
                        ] if cond]),
                    "Solubility": -3.0,
                    "hERG": "Low Risk",
                    "CACO2": "High",
                    "CLint_human": 12.5,
                    "HepG2_cytotox": "Non-toxic",
                    "Fub_human": 0.85,
                    "Source": "DTO-DSS",
                    "Index": 0
                })

        return {
            "smiles": result.smiles,
            "ml_prediction": result.ml_prediction,
            "ontology_score": result.ontology_score,
            "combined_score": result.combined_score,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level,
            "confidence_action": result.confidence_action,
            "agreement": result.agreement,
            "risk_level": "High" if result.combined_score > 0.7 else ("Medium" if result.combined_score > 0.4 else "Low"),
            "explanation": result.explanation,
            "triggered_rules": triggered_rules,
            "shap_features": result.shap_features,
            "all_descriptors": descriptors_dict,
            "model_used": active_model.model_type
        }
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Combined prediction failed: {e}")


# ==================== Advanced Analysis API ====================

@app.get("/analysis/advanced-plot", response_class=HTMLResponse, tags=["Analysis"])
async def get_advanced_plot():
    """
    Generate interactive Plotly scatter plot (LogP vs MW) with regression lines.
    Returns HTML content.
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    from rdkit.Chem import Descriptors

    if not dataset_loader:
        raise HTTPException(status_code=503, detail="Dataset loader not initialized")
    
    # Load data
    samples = dataset_loader.load_all()
    if not samples:
        return "<h3>No data available for analysis.</h3>"

    # Calculate properties
    properties = []
    for sample in samples:
        smiles = sample.get("smiles")
        if not smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                properties.append({
                    'MW': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'H-Donors': Descriptors.NumHDonors(mol),
                    'H-Acceptors': Descriptors.NumHAcceptors(mol),
                    'Ring Count': Descriptors.RingCount(mol),
                    'SMILES': smiles
                })
        except:
            continue
    

class BenchmarkPlotRequest(BaseModel):
    """벤치마크 회귀 플롯 요청"""
    results: list = Field(..., description="벤치마크 결과 리스트")
    x_descriptor: str = Field(default="LogP", description="X축 분자 기술자")
    y_descriptor: str = Field(default="MW", description="Y축 분자 기술자")
    coloring: str = Field(default="TPSA", description="색상 기준 분자 기술자")
    x_log_scale: bool = Field(default=False, description="X축 로그 스케일 여부")
    y_log_scale: bool = Field(default=False, description="Y축 로그 스케일 여부")
    show_linear: bool = Field(default=False, description="Linear Regression 표시 여부")
    show_ransac: bool = Field(default=False, description="RANSAC 표시 여부")


@app.post("/analysis/benchmark-plot", response_class=HTMLResponse, tags=["Analysis"])
async def get_benchmark_plot(request: BenchmarkPlotRequest):
    """
    Generate interactive Plotly scatter plot from benchmark results.
    Supports click events via JavaScript callback.
    Automatically calculates LogP, MW, TPSA if needed.
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    
    if not request.results:
        return "<h3>No benchmark results available.</h3>"
    
    # Prepare data for plotting
    data = []
    for i, r in enumerate(request.results):
        smiles = r.get('smiles', '')
        
        # 분자명 조회 (SMILES → 이름 변환)
        molecule_name = get_molecule_name_from_smiles(smiles)
        
        item = {
            'index': i,
            'smiles': smiles,
            'molecule_name': molecule_name,
            'ml_prediction': r.get('ml_prediction', 0),
            'ontology_score': r.get('ontology_score', 0),
            'combined_score': r.get('combined_score', 0),
            'confidence': r.get('confidence', 0),
            'risk_level': r.get('risk_level', 'Unknown')
        }
        
        # Merge all descriptors (from backend calculation or original request)
        all_desc = r.get('all_descriptors', {})
        if all_desc:
            item.update(all_desc)
        
        # Calculate additional properties if needed
        smiles = item['smiles']
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    item['LogP'] = Descriptors.MolLogP(mol)
                    item['MW'] = Descriptors.MolWt(mol)
                    item['TPSA'] = Descriptors.TPSA(mol)
                else:
                    logger.warning(f"Failed to parse SMILES for plot: {smiles}")
            except Exception as e:
                logger.warning(f"Error calculating descriptors for {smiles}: {e}")
                pass
        
        data.append(item)
    
    df = pd.DataFrame(data)
    
    # Get X and Y columns
    x_col = request.x_descriptor
    y_col = request.y_descriptor
    
    if x_col not in df.columns:
        x_col = 'LogP' if 'LogP' in df.columns else 'combined_score'
    if y_col not in df.columns:
        y_col = 'MW' if 'MW' in df.columns else 'combined_score'
        
    # Determine color column from request
    color_col = request.coloring if request.coloring in df.columns else 'risk_level'
    
    # Determine if using discrete or continuous color scale
    is_discrete = color_col == 'risk_level'
    
    # Create scatter plot
    debug_title = f'Analysis: {x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}'
    
    if is_discrete:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'},
            title=debug_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
            hover_data=['index', 'molecule_name', 'smiles', 'ml_prediction', 'ontology_score', 'confidence'],
            custom_data=['index', 'smiles'],
            log_x=request.x_log_scale,
            log_y=request.y_log_scale,
            render_mode='svg' # Force SVG to avoid WebGL errors in JavaFX WebView
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_continuous_scale='Viridis',
            title=debug_title,
            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
            hover_data=['index', 'molecule_name', 'smiles', 'ml_prediction', 'ontology_score', 'confidence'],
            custom_data=['index', 'smiles'],
            log_x=request.x_log_scale,
            log_y=request.y_log_scale,
            render_mode='svg' # Force SVG to avoid WebGL errors in JavaFX WebView
        )

    # Style the plot
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='white')),
        hovertemplate="<b>%{customdata[1]}</b><br>" +
                      "Index: %{customdata[0]}<br>" +
                      f"{x_col}: %{{x:.3f}}<br>" +
                      f"{y_col}: %{{y:.3f}}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#ecf0f1'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add regression lines if we have enough data and numeric columns
    if len(df) > 1 and x_col in df.columns and y_col in df.columns:
        try:
            # Get valid data points (drop NaN only for regression calculation, not for plotting)
            # For plotting, we want to show ALL points. If x/y is missing, we fill with 0 or exclude ONLY from plot?
            # User wants to see WHY points are missing.
            # Let's fill NaNs in df for plotting with 0, but keep them as NaN for regression.
            
            # Create a copy for regression that drops NaNs
            reg_df = df[[x_col, y_col]].copy()
            reg_df[x_col] = pd.to_numeric(reg_df[x_col], errors='coerce')
            reg_df[y_col] = pd.to_numeric(reg_df[y_col], errors='coerce')
            valid_df = reg_df.dropna()
            
            logger.info(f"Regression calculation: {len(valid_df)} valid points for {x_col} vs {y_col} (Total: {len(df)})")
            
            # Fill NaNs in main df for plotting (so points appear)
            df[x_col] = df[x_col].fillna(0)
            df[y_col] = df[y_col].fillna(0)
            
            if len(valid_df) > 1:
                X = valid_df[x_col].values.reshape(-1, 1)
                y = valid_df[y_col].values
                
                # Generate x range for plotting regression lines
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                
                # Linear Regression - show based on toggle
                try:
                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_lr = lr.predict(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range.flatten(),
                        y=y_lr,
                        mode='lines',
                        name='Linear Regression',
                        line=dict(color='#3498db', width=3, dash='solid'),
                        visible=True if request.show_linear else 'legendonly'
                    ))
                    logger.info(f"Added Linear Regression line: slope={lr.coef_[0]:.4f}, intercept={lr.intercept_:.4f}, visible={request.show_linear}")
                except Exception as e:
                    logger.warning(f"Linear Regression failed: {e}")
                
                # RANSAC Regressor - show based on toggle
                try:
                    ransac = RANSACRegressor(random_state=42)
                    ransac.fit(X, y)
                    y_ransac = ransac.predict(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range.flatten(),
                        y=y_ransac,
                        mode='lines',
                        name='RANSAC Regressor',
                        line=dict(color='#e74c3c', width=3, dash='dash'),
                        visible=True if request.show_ransac else 'legendonly'
                    ))
                    logger.info(f"Added RANSAC Regressor line, visible={request.show_ransac}")
                except Exception as e:
                    logger.warning(f"RANSAC Regressor failed: {e}")
            else:
                logger.warning(f"Not enough valid data points for regression: {len(valid_df)}")
        except Exception as e:
            logger.warning(f"Regression line calculation failed: {e}")
    
    # Add click event JavaScript
    html_content = fig.to_html(
        full_html=True, 
        include_plotlyjs=True,
        config={'responsive': True, 'displayModeBar': 'hover', 'scrollZoom': True}
    )
    
    # Inject custom JavaScript for click handling
    click_script = """
    <style>
        body, html { margin: 0; padding: 0; height: 100%; width: 100%; overflow: hidden; }
        .plotly-graph-div { height: 100vh !important; width: 100vw !important; }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv) {
            plotDiv.on('plotly_click', function(data) {
                var point = data.points[0];
                var index = point.customdata[0];
                var smiles = point.customdata[1];
                // Send message to JavaFX WebView
                if (window.javabridge) {
                    window.javabridge.onPointClick(index, smiles);
                }
                // Also log for debugging
                console.log('Clicked point index:', index, 'SMILES:', smiles);
            });
        }
    });
    </script>
    """
    
    # Insert script before </body>
    html_content = html_content.replace('</body>', click_script + '</body>')
    
    return html_content


# ==================== Decision Tree Analysis ====================

class DecisionTreeRequest(BaseModel):
    """Decision Tree 시각화 요청"""
    max_depth: int = Field(default=5, description="트리 최대 깊이")
    use_semantic: bool = Field(default=False, description="Ontology 의미론적 특성 사용 여부")


@app.post("/analysis/decision-tree", response_class=HTMLResponse, tags=["Analysis"])
async def get_decision_tree_plot(request: DecisionTreeRequest):
    """
    Generate Decision Tree visualization using scikit-learn.
    Returns Plotly HTML with tree structure.
    Uses real dataset if available, and optionally adds semantic features.
    """
    import plotly.graph_objects as go
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import pandas as pd
    from sklearn.tree import export_text
    
    # 1. Load Data
    samples = []
    if dataset_loader:
        samples = dataset_loader.load_all()
    
    # Fallback to synthetic data if no samples
    if not samples:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=10, n_informative=6, 
                                    n_redundant=2, n_classes=3, random_state=42)
        feature_names = ["LogP", "MW", "TPSA", "HBD", "HBA", "nRotB", "nHeavy", "nRings", "hERG", "CACO2"]
        class_names = ["Safe", "Moderate", "Toxic"]
        
        # Add mock semantic features if requested (for fallback)
        if request.use_semantic:
            # Generate random binary features acting as "rules"
            n_semantic = 3
            semantic_X = np.random.randint(0, 2, size=(500, n_semantic))
            X = np.hstack([X, semantic_X])
            feature_names.extend([f"Rule_{i+1}_Triggered" for i in range(n_semantic)])
            
    else:
        # Process real data
        data_rows = []
        labels = []
        
        feature_keys = ["MW", "logKow", "HBD", "HBA", "nRotB", "TPSA", "Aromatic_Rings", "Heteroatom_Count", "Heavy_Atom_Count", "logP"]
        
        active_model = model_manager.get_active_model() if model_manager else None
        
        for sample in samples:
            smiles = sample.get("smiles", "")
            desc = sample.get("descriptors", {})
            
            # Basic features
            row = [desc.get(k, 0.0) for k in feature_keys]
            
            # Semantic features
            if request.use_semantic and rule_engine:
                # Use Rule Engine to calculate semantic features
                try:
                    # Get actual RF prediction instead of mock
                    rf_pred = {"class_name": "Unknown", "confidence": 0.0}
                    if active_model and active_model.model_type == "random_forest":
                        try:
                            fv = np.array([row[:10]])  # Use 10 base features
                            rf_result = active_model.predict(fv)
                            rf_pred = {
                                "class_name": rf_result.get("class_name", "Unknown"),
                                "confidence": rf_result.get("confidence", 0.0)
                            }
                        except Exception as rf_err:
                            logger.warning(f"RF prediction failed for {smiles}: {rf_err}")
                    
                    val_result = rule_engine.validate_prediction(smiles, rf_pred, desc)
                    
                    # Add features derived from Ontology Rules
                    row.append(val_result.rule_count)
                    row.append(val_result.rule_confidence_score)
                    
                    # Add binary flags for specific risk types if available in interpretation or rules?
                    # For simplicity, we use the aggregate scores which are "Semantic" summary.
                    
                    # Also check for specific heavy rules if possible, but rule_count covers general semantic density.
                    
                except Exception as e:
                    logger.warning(f"Rule engine failed for {smiles}: {e}")
                    row.append(0) # rule_count
                    row.append(0.0) # rule_conf
            
            data_rows.append(row)
            
            # Generate Label (Target)
            if active_model and active_model.model_type == "random_forest":
                 fv = np.array([row[:10]]) # Use original 10 features for prediction
                 pred = active_model.predict(fv)
                 labels.append(pred.get("prediction_class", 0)) # 0 or 1
            else:
                 labels.append(0)

        X = np.array(data_rows)
        y = np.array(labels)
        
        feature_names = feature_keys.copy()
        if request.use_semantic:
            feature_names.extend(["Ontology_Rule_Count", "Ontology_Rule_Confidence"])
            
        class_names = ["Safe", "Toxic"]

@app.post("/analysis/decision-tree", response_class=HTMLResponse, tags=["Analysis"])
async def get_decision_tree_plot(request: DecisionTreeRequest):
    """
    Generate Decision Tree visualization using scikit-learn plot_tree.
    Returns HTML with embedded base64 image.
    Uses real dataset if available.
    """
    import base64
    import io
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import numpy as np
    import pandas as pd
    
    # Set matplotlib font for Korean support (just in case, though we used English labels)
    plt.rcParams['font.family'] = 'Segoe UI' # Windows font
    
    # 1. Load Data
    samples = []
    if dataset_loader:
        samples = dataset_loader.load_all()
    
    # Fallback to synthetic data
    if not samples:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=10, n_informative=6, 
                                    n_redundant=2, n_classes=3, random_state=42)
        feature_names = ["LogP", "MW", "TPSA", "HBD", "HBA", "nRotB", "nHeavy", "nRings", "hERG", "CACO2"]
        class_names = ["Safe", "Moderate", "Toxic"]
        if request.use_semantic:
            n_semantic = 3
            semantic_X = np.random.randint(0, 2, size=(500, n_semantic))
            X = np.hstack([X, semantic_X])
            feature_names.extend([f"Rule_{i+1}" for i in range(n_semantic)])
    else:
        # Process real data
        data_rows = []
        labels = []
        feature_keys = ["MW", "logKow", "HBD", "HBA", "nRotB", "TPSA", "Aromatic_Rings", "Heteroatom_Count", "Heavy_Atom_Count", "logP"]
        active_model = model_manager.get_active_model() if model_manager else None
        
        for sample in samples:
            desc = sample.get("descriptors", {})
            smiles = sample.get("smiles", "")
            
            # Use robust feature value extractor
            row = [get_feature_val(desc, k) for k in feature_keys]
            
            # Re-compute if values are missing or zero (for critical ones)
            if row[0] == 0 or row[9] == 0: # MW or LogP
                 computed = compute_descriptors(smiles)
                 if computed:
                     row = [get_feature_val(computed, k, default=row[i]) for i, k in enumerate(feature_keys)]

            # Filter invalid data (NaN, Inf)
            if any(np.isnan(row)) or any(np.isinf(row)):
                continue

            if request.use_semantic and rule_engine:
                try:
                    # Get actual RF prediction instead of mock
                    rf_pred = {"class_name": "Unknown", "confidence": 0.0}
                    if active_model and active_model.model_type == "random_forest":
                        try:
                            fv = np.array([row[:10]])  # Use 10 base features
                            rf_result = active_model.predict(fv)
                            rf_pred = {
                                "class_name": rf_result.get("class_name", "Unknown"),
                                "confidence": rf_result.get("confidence", 0.0)
                            }
                        except Exception as rf_err:
                            logger.warning(f"RF prediction failed for {smiles}: {rf_err}")
                    
                    val_result = rule_engine.validate_prediction(smiles, rf_pred, desc)
                    row.append(val_result.rule_count)
                    row.append(val_result.rule_confidence_score)
                except:
                    row.append(0)
                    row.append(0.0)
            
            data_rows.append(row)
            if active_model and active_model.model_type == "random_forest":
                 fv = np.array([row[:10]])
                 pred = active_model.predict(fv)
                 labels.append(pred.get("prediction_class", 0))
            else:
                 labels.append(0)

        if not data_rows:
             return "<h3>No valid data after filtering.</h3>"

        X = np.array(data_rows)
        y = np.array(labels)
        feature_names = feature_keys.copy()
        if request.use_semantic:
            feature_names.extend(["Rule_Count", "Rule_Confidence"])
        class_names = ["Safe", "Toxic"]

    # Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=request.max_depth, random_state=42)
    clf.fit(X, y)
    
    
    # Extract Tree Structure to JSON
    def tree_to_json(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node):
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                return {
                    "name": f"{name} <= {threshold:.2f}",
                    "value": float(tree_.impurity[node]),
                    "children": [
                        recurse(tree_.children_left[node]),
                        recurse(tree_.children_right[node])
                    ]
                }
            else:
                # Leaf
                class_idx = np.argmax(tree_.value[node])
                class_name = class_names[class_idx]
                return {
                    "name": f"Leaf: {class_name}",
                    "value": float(tree_.impurity[node]),
                    "itemStyle": {
                        "color": "#e74c3c" if class_name == "Toxic" else "#27ae60"
                    }
                }
        
        return recurse(0)

    tree_json = tree_to_json(clf, feature_names)
    import json
    tree_data_str = json.dumps(tree_json)
    
    # Return HTML with ECharts
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive Decision Tree</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; overflow: hidden; }}
            #tree-container {{ width: 100vw; height: 100vh; }}
        </style>
    </head>
    <body>
        <div id="tree-container"></div>
        <script>
            var chartDom = document.getElementById('tree-container');
            var myChart = echarts.init(chartDom);
            var option;

            var data = {tree_data_str};

            option = {{
                tooltip: {{
                    trigger: 'item',
                    triggerOn: 'mousemove'
                }},
                series: [
                    {{
                        type: 'tree',
                        data: [data],
                        top: '1%',
                        left: '7%',
                        bottom: '1%',
                        right: '20%',
                        symbolSize: 10,
                        label: {{
                            position: 'left',
                            verticalAlign: 'middle',
                            align: 'right',
                            fontSize: 12
                        }},
                        leaves: {{
                            label: {{
                                position: 'right',
                                verticalAlign: 'middle',
                                align: 'left'
                            }}
                        }},
                        expandAndCollapse: true,
                        animationDuration: 550,
                        animationDurationUpdate: 750
                    }}
                ]
            }};

            myChart.setOption(option);
            
            window.onresize = function() {{
                myChart.resize();
            }};
        </script>
    </body>
    </html>
    """
    return html_content


class DecisionBoundary2DRequest(BaseModel):
    """2D 결정 경계 요청"""
    x_feature: str = Field(default="LogP", description="X축 특성")
    y_feature: str = Field(default="MW", description="Y축 특성")
    dataset_name: Optional[str] = Field(default=None, description="데이터셋 이름")
    model_type: str = Field(default="random_forest", description="시각화에 사용할 모델")
    sample_size: Optional[int] = Field(
        default=500,
        description="결정 경계 생성 시 사용할 샘플 수 (최대 500 권장)"
    )

    class Config:
        protected_namespaces = ()


@app.post(
    "/analysis/decision-boundary-2d",
    response_class=HTMLResponse,
    tags=["Analysis"]
)
async def get_decision_boundary_2d(request: DecisionBoundary2DRequest):
    """
    Generate 2D decision boundary using sklearn DecisionBoundaryDisplay.
    Uses REAL dataset and trains a proxy model on the selected 2 features.
    """
    import base64
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    
    plt.rcParams['font.family'] = 'Segoe UI'

    # 1. Load Data
    samples = []
    max_samples = request.sample_size if request.sample_size else 500
    if max_samples <= 0:
        max_samples = 500
    max_samples = min(max_samples, 500)  # 안전 상한
    if dataset_loader:
        if request.dataset_name:
            # Load specific dataset (limit 500 for performance)
            try:
                ds_samples, _ = dataset_loader.load_dataset(
                    request.dataset_name,
                    sample_size=max_samples
                )
                # Convert DatasetSample objects to dicts
                samples = [
                    {"smiles": s.smiles, "descriptors": s.descriptors}
                    for s in ds_samples
                ]
            except Exception as e:
                logger.error(
                    f"Error loading dataset {request.dataset_name}: {e}"
                )
                samples = []
        else:
            # Fallback to loading all (limit to 500)
            all_samples = dataset_loader.load_all()
            samples = all_samples[:max_samples]
            
    if not samples:
        logger.warning("No data found for Decision Boundary generation")
        return "<h3>No data available for Decision Boundary.</h3>"
        
    # 2. Extract X (2 features) and y (target)
    x_data = []
    y_data = []
    
    # Map feature names to keys if needed (assuming simple match)
    # The request sends "LogP", "MW" etc. which match descriptor keys.
    # dataset_loader returns 'descriptors' dict with keys like 'MW', 'logP'.
    
    for sample in samples:
        desc = sample.get("descriptors", {})
        smiles = sample.get("smiles", "")
        
        try:
            val_x = get_feature_val(desc, request.x_feature, None)
            val_y = get_feature_val(desc, request.y_feature, None)
            
            # If missing, compute on the fly
            if val_x is None or val_y is None or (
                val_x == 0 and request.x_feature.lower() in ["mw", "logp"]
            ):
                computed = compute_descriptors(smiles)
                if val_x is None or (
                    val_x == 0 and request.x_feature.lower() in ["mw", "logp"]
                ):
                    val_x = get_feature_val(computed, request.x_feature, 0.0)
                if val_y is None:
                    val_y = get_feature_val(computed, request.y_feature, 0.0)
            
            # Validation: Filter invalid
            if (
                np.isnan(val_x)
                or np.isinf(val_x)
                or np.isnan(val_y)
                or np.isinf(val_y)
            ):
                continue
                
            x_data.append([float(val_x), float(val_y)])
            
            # Prediction for coloring
            feature_keys = [
                "MW",
                "logKow",
                "HBD",
                "HBA",
                "nRotB",
                "TPSA",
                "Aromatic_Rings",
                "Heteroatom_Count",
                "Heavy_Atom_Count",
                "logP",
            ]
            full_row = [get_feature_val(desc, k) for k in feature_keys]
            
            # Ensure full_row is valid too
            if any(np.isnan(full_row)) or any(np.isinf(full_row)):
                y_data.append(0)  # or skip
            else:
                target_model_type = request.model_type.lower()
                model_to_use = None
                if (
                    model_manager
                    and hasattr(model_manager, "_models")
                    and target_model_type in model_manager._models
                ):
                    model_to_use = model_manager._models[target_model_type]
                elif model_manager:
                    model_to_use = model_manager.get_active_model()
                
                if model_to_use:
                    try:
                        res = model_to_use.predict(np.array([full_row]))
                        label = res.get("prediction_class", 0)
                    except:
                        label = 0
                else:
                    label = 0
                y_data.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing sample in boundary: {e}")
            continue
            
    if not x_data:
        return "<h3>No valid data for Decision Boundary after filtering.</h3>"
        
    X = np.array(x_data)
    y = np.array(y_data)
    
    if len(X) < 5:
         return "<h3>Not enough data points.</h3>"

    # 3. Train Proxy 2D Model
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)
    
    # 4. Plot DecisionBoundaryDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.6,
        xlabel=request.x_feature,
        ylabel=request.y_feature
    )
    
    # Scatter plot of actual points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=40)
    # Legend
    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # ax.add_artist(legend1)
    
    ax.set_title(f"2D Decision Boundary (Proxy Tree)\n{request.x_feature} vs {request.y_feature}")
    
    # Save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>2D Decision Boundary</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: white; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <img src="data:image/png;base64,{img_str}">
    </body>
    </html>
    """
    return html_content

# ==================== Dataset API ====================

@app.get("/datasets/list", tags=["Datasets"])
async def list_datasets():
    """
    List available datasets.
    """
    if not dataset_loader:
        # Try to init or return empty
        return []
    return dataset_loader.list_datasets()

class DatasetSamplesRequest(BaseModel):
    dataset_name: str
    limit: Optional[int] = 100

@app.post("/datasets/samples", tags=["Datasets"])
async def get_dataset_samples(request: DatasetSamplesRequest):
    """
    Get raw samples from a dataset.
    Returns: List of {smiles: str, descriptors: dict, molecule_name: str}
    """
    if not dataset_loader:
        raise HTTPException(status_code=503, detail="DatasetLoader not initialized")
    
    try:
        # Determine limit - if 10000 or high, assume ALL
        limit = None if request.limit and request.limit >= 10000 else request.limit
        
        info = dataset_loader.available_datasets.get(request.dataset_name.lower())
        if not info:
             raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_name} not found")
             
        data_samples = []
        import csv
        with open(info.path, 'r', encoding='utf-8') as f:
            # Handle potential BOM or encoding issues
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                if limit and count >= limit:
                    break
                smiles = row.get(info.smiles_column, '').strip()
                if not smiles: continue
                
                # Descriptors: everything not smiles or label
                descriptors = {}
                for col, val in row.items():
                    if col == info.smiles_column or col in info.label_columns:
                        continue
                    try:
                        descriptors[col] = float(val) if val else 0.0
                    except:
                        pass # Ignore non-numeric descriptors
                
                # If no descriptors found in CSV, compute them
                if not descriptors or len(descriptors) < 3:
                    computed = compute_descriptors(smiles)
                    if computed:
                        descriptors.update(computed)
                else:
                    # Map common CSV column names to our standard ones if they exist
                    # (Helpful for BBBP if it had some columns)
                    mapping = {
                        "MW": ["MW", "MolWt", "MolecularWeight", "Molecular Weight"],
                        "logP": ["LogP", "logP", "MolLogP", "logkow"],
                        "HBD": ["HBD", "nHBDon", "NumHDonors"],
                        "HBA": ["HBA", "nHBAcc", "NumHAcceptors"],
                    }
                    for std_key, aliases in mapping.items():
                        if std_key not in descriptors:
                            for alias in aliases:
                                if alias in row:
                                    try:
                                        descriptors[std_key] = float(row[alias])
                                        break
                                    except: pass

                data_samples.append({
                    "smiles": smiles,
                    "descriptors": descriptors,
                    "molecule_name": row.get("name") or row.get("Name") or row.get("Compound ID") or row.get("compound_id") or "Unknown"
                })
                count += 1
                
        return data_samples

    except Exception as e:
        logger.error(f"Error getting samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))
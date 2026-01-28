# SDSS-Tox: Semantic Decision Support System for Toxicity Prediction

Drug Target Ontology     

##  

SDSS-Tox Machine Learning     .

- :      
-  : Java (GUI/), Python (ML ), RDF/OWL
- :  AI (XAI),  ,  

---

##  

```
SDSS-Tox/

    (6)
    README.md                  #  
    pom.xml                    # Maven Java 
    requirements.txt           # Python 
    .gitignore, .classpath, .project

 Frontend (JavaFX)
    src/main/java/com/example/dto/
        Main.java                    #  
        core/                        #  
           DtoLoader.java          # OWL 
           DtoQuery.java           # /
           OntologyValidator.java  # ML 
           ChemicalMapper.java     #  
        gui/                         # GUI 
        api/                         # API 
        utils/                       # 
        data/                        #  
        visualization/               # 

 Backend (FastAPI)
    backend/app/
        main.py                # FastAPI 
        config/                #  
           ontology_rules.yaml
        services/
            model_manager.py
            base_model.py
            simple_qsar.py
            predictors/        # ML 
               rf_predictor.py
               dt_predictor.py
               sdt_predictor.py
               pytorch_predictor.py
               ensemble_dss.py
               combined_predictor.py
               semantic_decision_tree.py
            ontology/          #  
               dto_parser.py
               dto_rule_engine.py
               read_across.py
            explainability/    # 
               shap_explainer.py
            data/              # 
                dataset_loader.py

 
    bbbp/           # BBBP (Blood-Brain Barrier)
    esol/           # ESOL ()
    qm7/, qm8/, qm9/ #   
    clintox/        # ClinTox
    sider/          # SIDER ()
    tox21/          # Tox21
    muv/            # MUV
    hiv/            # HIV
    lipophilicity/  # 
    freesolv/       # FreeSolv
    ontology/       #  
        dto.rdf     # Drug Target Ontology

 
    run_dss.py                    #   (Java/Python)
    write_fxml.py                 # FXML 
    analysis/
       run_batch_analysis.py     #  
       run_bbbp_analysis.py      # BBBP 
       advanced_analysis.py      #  
       debug_backend_logic.py    # 
    demos/
        streamlit_sdt_dashboard.py
        streamlit_hf_sdt.py

 
    backend/tests/
        test_integration.py
        test_rf_predictor.py
        test_shap_explainer.py
        test_simple_qsar.py

   
    results/                  #  
    backend/models/          #   (.pkl)

 IDE/Git 
     .git/                    # Git 
     .github/                 # GitHub Actions
     .venv/                   # Python 
     .settings/               # Eclipse 
     .vscode/                 # VS Code 
```

---

##  

###  
```bash
# Java 11+
java -version

# Python 3.8+
python --version

# Maven (  )
mvn --version
```

###  

Python 
```bash
#    
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

#  
pip install -r requirements.txt
```

Java 
```bash
# Maven 
mvn clean package

#  IDE  (Eclipse/IntelliJ)
```

###  

**   ()**

Windows ( ):
```bash
run.bat
```

PowerShell:
```powershell
.\run.ps1
```

Python:
```bash
python run.py
```

** **

Python  :
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Java GUI :
```bash
mvn clean javafx:run
```

###   

BBBP  
```bash
python scripts/analysis/run_bbbp_analysis.py
```

 
```bash
python scripts/analysis/run_batch_analysis.py
```

Streamlit  ()
```bash
streamlit run scripts/demos/streamlit_sdt_dashboard.py
```

---

##   

### data/ontology/
- : DTO (Drug Target Ontology) 
- : dto.rdf (RDF/OWL )
- : 
  -   
  - ML  
  -  

### backend/app/services/
Python    :

|  |  |
|------|------|
| predictors/ | RF, DT, SDT, PyTorch  ML  |
| ontology/ | DTO ,  ,   |
| explainability/ | SHAP    |
| data/ |   |

### scripts/
- analysis/:      
- demos/: Streamlit   
- run_dss.py: Java/Python  

### src/main/java/com/example/dto/
Java   :

|  |  |
|--------|------|
| core/ | DtoLoader, DtoQuery,   |
| gui/ | JavaFX    |
| api/ | Python  API  |
| utils/ |   (  ) |
| visualization/ |     |

---

## 

###  
```
GUI/Backend -> DtoLoader (data/ontology/dto.rdf) -> OWL API
```

###  
```
SMILES  -> RDKit ( ) -> ML  
```

###  
```
ML  -> OntologyValidator -> DTO  ->  
```

###  
```
Combined Score, Confidence, Rule Triggers, SHAP  
```

---

##  

###   
```python
# backend/app/services/predictors/your_model.py
from backend.app.services.base_model import BaseModel

class YourPredictor(BaseModel):
    def predict(self, smiles: str) -> float:
        # 
        pass
```

###   
- backend/app/config/ontology_rules.yaml 
- backend/app/services/ontology/dto_rule_engine.py 

### GUI 
- src/main/java/com/example/dto/gui/   
- src/main/resources/main.fxml 

---

##  

### Backend (Java)
- : JavaFX (GUI)
- : OWL API, RDF
- : Maven
- JDK: 11+

### Backend (Python)
- API: FastAPI, Uvicorn
- ML: scikit-learn, PyTorch, XGBoost
- : rdflib
- : SHAP
- : RDKit

### 
- : CSV, RDF/OWL
- : MoleculeNet, ChEMBL, DTO

---

##   

###  
```bash
cd backend/tests
pytest test_integration.py
pytest test_rf_predictor.py
```

### 
```bash
python scripts/analysis/run_batch_analysis.py
```

---

##  

###   
```
Error: "Failed to load ontology"
Solution: 
  - dto.rdf  : data/ontology/dto.rdf
  -   : RDF/OWL
```

### Python  
```bash
#   
pip install -r requirements.txt --upgrade
```

### Java  
```bash
#  
mvn clean
#   
mvn dependency:resolve
```

---

##   

###  
- Java:    (core/, gui/, api/, utils/)
- Python:    (predictors/, ontology/, explainability/)
-  : 15 -> 6  (60% )

###  
-  : data/ontology/ 
-  : scripts/  
-  : backend/app/config/ 

###  
- import  
-   
-  

---

##   

-  : GitHub Issues
- :   
- :    

---

## 

[ ]

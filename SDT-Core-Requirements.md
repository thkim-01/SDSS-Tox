# SDT-Core: Semantic Decision Tree Engine Development Project

## Project Overview
- **Project Name**: SDT-Core
- **Purpose**: Development of an independent Semantic Decision Tree engine
- **Final Goal**: Design as a plugin that can be integrated into the SDSS-Tox project

## Core Requirements

### 1. Project Structure
```
SDT-Core/
├── sdt/
│   ├── __init__.py
│   ├── core/
│   │   ├── decision_tree.py      # Core Decision Tree engine
│   │   ├── semantic_layer.py     # Semantic layer (ontology integration)
│   │   └── rule_engine.py        # Rule-based inference engine
│   ├── utils/
│   │   ├── ontology_loader.py    # RDF/OWL ontology loader
│   │   └── feature_extractor.py  # Feature extraction
│   └── explainability/
│       └── explainer.py          # Explainability module
├── tests/
├── examples/
├── requirements.txt
├── setup.py
└── README.md
```

### 2. Tech Stack
- **Python 3.8+**
- **Ontology**: rdflib (RDF/OWL processing)
- **ML**: scikit-learn, numpy
- **Visualization**: graphviz (decision tree visualization)

### 3. Core Features

#### A. SemanticDecisionTree Class
```python
class SemanticDecisionTree:
    def __init__(self, ontology_path=None, rules_path=None):
        """
        Decision tree based on ontology and rules
        """
        
    def fit(self, X, y, feature_names=None, semantic_constraints=None):
        """
        Train tree considering semantic constraints
        """
        
    def predict(self, X):
        """
        Prediction + semantic validation
        """
        
    def explain(self, X, output_format='text'):
        """
        Generate decision path and ontology-based explanations
        output_format: 'text', 'json', 'graph'
        """
```

#### B. Ontology Integration
- Load RDF/OWL files (compatible with existing dto.rdf)
- SPARQL query support
- Ontology-based feature relationship inference

#### C. Rule Engine
- YAML format rule definition (compatible with existing ontology_rules.yaml)
- IF-THEN rule application
- Conflict resolution mechanism

### 4. SDSS-Tox Integration Interface

**Required API Design**:
```python
# Should be usable in SDSS-Tox like this
from sdt import SemanticDecisionTree

# Initialize
sdt = SemanticDecisionTree(
    ontology_path='../SDSS-Tox/data/ontology/dto.rdf',
    rules_path='../SDSS-Tox/backend/app/config/ontology_rules.yaml'
)

# Train
sdt.fit(X_train, y_train, feature_names=features)

# Predict + Explain
predictions = sdt.predict(X_test)
explanations = sdt.explain(X_test[0])
```

### 5. Output Format

**Explainability Output Example**:
```json
{
  "prediction": "toxic",
  "confidence": 0.87,
  "decision_path": [
    {"feature": "molecular_weight", "threshold": 450, "value": 523, "direction": "right"},
    {"feature": "logP", "threshold": 3.5, "value": 4.2, "direction": "right"}
  ],
  "semantic_reasoning": [
    "High molecular weight indicates poor bioavailability",
    "Lipophilicity (logP > 3.5) suggests potential toxicity"
  ],
  "ontology_concepts": ["Toxicity", "Lipophilicity", "Bioavailability"]
}
```

### 6. Testing Requirements
- Unit tests (pytest)
- Integration tests (using SDSS-Tox data)
- Performance benchmarks

### 7. Documentation
- API documentation (docstrings)
- Usage examples (examples/)
- Integration guide (how to integrate with SDSS-Tox)

## Reference Data
- **Ontology**: `../SDSS-Tox/data/ontology/dto.rdf`
- **Rules File**: `../SDSS-Tox/backend/app/config/ontology_rules.yaml`
- **Sample Data**: `../SDSS-Tox/data/tox21/tox21.csv`

## Success Criteria
1. Independent pip package that can run standalone
2. Importable into SDSS-Tox project
3. Support for ontology-based reasoning
4. Explainable prediction results
5. 90%+ test coverage

## Getting Started

**Initial Command**:
"Please build the SDT-Core project according to the above specifications. Start by creating the project structure."

## Integration with SDSS-Tox

After SDT-Core is completed, it should be integrated into SDSS-Tox as follows:

### Installation
```bash
cd SDT-Core
pip install -e .
```

### Usage in SDSS-Tox
```python
# In SDSS-Tox backend
from sdt import SemanticDecisionTree

# Initialize with SDSS-Tox resources
sdt = SemanticDecisionTree(
    ontology_path='data/ontology/dto.rdf',
    rules_path='backend/app/config/ontology_rules.yaml'
)

# Use in prediction pipeline
predictions = sdt.fit(X_train, y_train).predict(X_test)
explanations = sdt.explain(X_test)
```

## Expected Deliverables
1. Complete SDT-Core package
2. Documentation and examples
3. Test suite with high coverage
4. Integration guide for SDSS-Tox
5. Performance benchmarks

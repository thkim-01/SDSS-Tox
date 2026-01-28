# SDSS-Tox: Semantic Decision Support System for Toxicity Prediction

Drug Target Ontology 기반 의약품 독성 예측 시스템

## 프로젝트 개요

SDSS-Tox는 Machine Learning과 온톨로지를 결합한 신경기호 의사결정 시스템입니다.

- 목표: 약물의 독성을 예측하고 설명가능한 결과 제공
- 기술 스택: Java (GUI/온톨로지), Python (ML 백엔드), RDF/OWL
- 특징: 설명가능한 AI (XAI), 앙상블 모델, 의미론적 검증

---

## 프로젝트 구조

```
SDSS-Tox/
│
├── 루트 설정 파일 (6개)
│   ├── README.md                  # 이 파일
│   ├── pom.xml                    # Maven Java 빌드
│   ├── requirements.txt           # Python 의존성
│   └── .gitignore, .classpath, .project
│
├── Frontend (JavaFX)
│   └── src/main/java/com/example/dto/
│       ├── Main.java                    # 메인 엔트리
│       ├── core/                        # 온톨로지 처리
│       │   ├── DtoLoader.java          # OWL 로드
│       │   ├── DtoQuery.java           # 검색/쿼리
│       │   ├── OntologyValidator.java  # ML 검증
│       │   └── ChemicalMapper.java     # 화학물질 매핑
│       ├── gui/                         # GUI 컨트롤러
│       ├── api/                         # API 클라이언트
│       ├── utils/                       # 유틸리티
│       ├── data/                        # 데이터 로더
│       └── visualization/               # 시각화
│
├── Backend (FastAPI)
│   └── backend/app/
│       ├── main.py                # FastAPI 진입점
│       ├── config/                # 설정 파일
│       │   └── ontology_rules.yaml
│       └── services/
│           ├── model_manager.py
│           ├── base_model.py
│           ├── simple_qsar.py
│           ├── predictors/        # ML 모델
│           │   ├── rf_predictor.py
│           │   ├── dt_predictor.py
│           │   ├── sdt_predictor.py
│           │   ├── pytorch_predictor.py
│           │   ├── ensemble_dss.py
│           │   ├── combined_predictor.py
│           │   └── semantic_decision_tree.py
│           ├── ontology/          # 온톨로지 처리
│           │   ├── dto_parser.py
│           │   ├── dto_rule_engine.py
│           │   └── read_across.py
│           ├── explainability/    # 해석성
│           │   └── shap_explainer.py
│           └── data/              # 데이터
│               └── dataset_loader.py
│
├── 데이터
│   ├── bbbp/           # BBBP (Blood-Brain Barrier)
│   ├── esol/           # ESOL (용해도)
│   ├── qm7/, qm8/, qm9/ # 양자 기계 특성
│   ├── clintox/        # ClinTox
│   ├── sider/          # SIDER (부작용)
│   ├── tox21/          # Tox21
│   ├── muv/            # MUV
│   ├── hiv/            # HIV
│   ├── lipophilicity/  # 친지성
│   ├── freesolv/       # FreeSolv
│   └── ontology/       # 온톨로지 데이터
│       └── dto.rdf     # Drug Target Ontology
│
├── 스크립트
│   ├── run_dss.py                    # 메인 실행 (Java/Python)
│   ├── write_fxml.py                 # FXML 생성
│   ├── analysis/
│   │   ├── run_batch_analysis.py     # 배치 분석
│   │   ├── run_bbbp_analysis.py      # BBBP 분석
│   │   ├── advanced_analysis.py      # 고급 분석
│   │   └── debug_backend_logic.py    # 디버깅
│   └── demos/
│       ├── streamlit_sdt_dashboard.py
│       └── streamlit_hf_sdt.py
│
├── 테스트
│   └── backend/tests/
│       ├── test_integration.py
│       ├── test_rf_predictor.py
│       ├── test_shap_explainer.py
│       └── test_simple_qsar.py
│
├── 결과 및 모델
│   ├── results/                  # 분석 결과
│   └── backend/models/          # 학습된 모델 (.pkl)
│
└── IDE/Git 설정
    ├── .git/                    # Git 저장소
    ├── .github/                 # GitHub Actions
    ├── .venv/                   # Python 가상환경
    ├── .settings/               # Eclipse 설정
    └── .vscode/                 # VS Code 설정
```

---

## 빠른 시작

### 사전 요구사항
```bash
# Java 11+
java -version

# Python 3.8+
python --version

# Maven (또는 번들 사용)
mvn --version
```

### 환경 설정

Python 환경
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

Java 빌드
```bash
# Maven으로 빌드
mvn clean package

# 또는 IDE에서 빌드 (Eclipse/IntelliJ)
```

### 애플리케이션 실행

**가장 간단한 실행 (권장)**

Windows (배치 파일):
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

**개별 실행**

Python 백엔드만 실행:
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Java GUI만 실행:
```bash
mvn clean javafx:run
```

### 데이터 분석 스크립트

BBBP 데이터셋 분석
```bash
python scripts/analysis/run_bbbp_analysis.py
```

배치 분석
```bash
python scripts/analysis/run_batch_analysis.py
```

Streamlit 대시보드 (데모)
```bash
streamlit run scripts/demos/streamlit_sdt_dashboard.py
```

---

## 주요 디렉토리 설명

### data/ontology/
- 용도: DTO (Drug Target Ontology) 관리
- 파일: dto.rdf (RDF/OWL 형식)
- 역할: 
  - 온톨로지 기반 추론
  - ML 모델 검증
  - 설명가능성 향상

### backend/app/services/
Python 서비스 계층의 역할별 구조:

| 폴더 | 역할 |
|------|------|
| predictors/ | RF, DT, SDT, PyTorch 등 ML 모델 |
| ontology/ | DTO 파싱, 규칙 엔진, 유사성 분석 |
| explainability/ | SHAP 기반 모델 해석 |
| data/ | 데이터셋 로더 |

### scripts/
- analysis/: 데이터 분석 및 모델 평가 스크립트
- demos/: Streamlit 기반 상호작용 대시보드
- run_dss.py: Java/Python 통합 실행

### src/main/java/com/example/dto/
Java 애플리케이션의 역할별 구조:

| 패키지 | 역할 |
|--------|------|
| core/ | DtoLoader, DtoQuery, 온톨로지 처리 |
| gui/ | JavaFX 컨트롤러 및 윈도우 |
| api/ | Python 백엔드 API 클라이언트 |
| utils/ | 범용 유틸리티 (테이블 편집 등) |
| visualization/ | 차트 및 그래프 렌더링 |

---

## 워크플로우

### 온톨로지 로딩
```
GUI/Backend -> DtoLoader (data/ontology/dto.rdf) -> OWL API
```

### 분자 분석
```
SMILES 입력 -> RDKit (분자 서술자) -> ML 모델 예측
```

### 의미론적 검증
```
ML 예측 -> OntologyValidator -> DTO 검색 -> 설명 생성
```

### 결과 표시
```
Combined Score, Confidence, Rule Triggers, SHAP 해석 등
```

---

## 개발 가이드

### 새 모델 추가
```python
# backend/app/services/predictors/your_model.py
from backend.app.services.base_model import BaseModel

class YourPredictor(BaseModel):
    def predict(self, smiles: str) -> float:
        # 구현
        pass
```

### 온톨로지 규칙 추가
- backend/app/config/ontology_rules.yaml 수정
- backend/app/services/ontology/dto_rule_engine.py 업데이트

### GUI 확장
- src/main/java/com/example/dto/gui/ 에 컨트롤러 추가
- src/main/resources/main.fxml 업데이트

---

## 기술 스택

### Backend (Java)
- 프레임워크: JavaFX (GUI)
- 온톨로지: OWL API, RDF
- 빌드: Maven
- JDK: 11+

### Backend (Python)
- API: FastAPI, Uvicorn
- ML: scikit-learn, PyTorch, XGBoost
- 온톨로지: rdflib
- 설명가능성: SHAP
- 분자: RDKit

### 데이터
- 포맷: CSV, RDF/OWL
- 출처: MoleculeNet, ChEMBL, DTO

---

## 성능 및 검증

### 테스트 실행
```bash
cd backend/tests
pytest test_integration.py
pytest test_rf_predictor.py
```

### 벤치마크
```bash
python scripts/analysis/run_batch_analysis.py
```

---

## 문제 해결

### 온톨로지 로드 실패
```
Error: "Failed to load ontology"
Solution: 
  - dto.rdf 경로 확인: data/ontology/dto.rdf
  - 파일 형식 확인: RDF/OWL
```

### Python 패키지 누락
```bash
# 모든 의존성 설치
pip install -r requirements.txt --upgrade
```

### Java 빌드 오류
```bash
# 캐시 삭제
mvn clean
# 모든 의존성 재다운로드
mvn dependency:resolve
```

---

## 주요 수정 사항

### 구조 개선
- Java: 역할별 패키지 분리 (core/, gui/, api/, utils/)
- Python: 기능별 서비스 분리 (predictors/, ontology/, explainability/)
- 루트 정리: 15개 -> 6개 파일 (60% 감소)

### 파일 정리
- 온톨로지 관리: data/ontology/ 통합
- 스크립트 관리: scripts/ 폴더 체계화
- 설정 파일: backend/app/config/ 중앙화

### 코드 정리
- import 경로 업데이트
- 임시 파일 제거
- 문서화 개선

---

## 연락처 및 지원

- 이슈 보고: GitHub Issues
- 문의: 팀 리더에게 연락
- 문서: 각 폴더별 설명 참조

---

## 라이선스

[라이선스 정보]

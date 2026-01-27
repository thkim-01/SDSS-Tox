import pandas as pd
import os
from backend.semantic_decision_tree import SemanticDecisionTree, TargetFisher, DTOKnowledgeBase
from tqdm import tqdm

def run_bbbp_analysis():
    print(">>> [System] BBBP 데이터셋 로딩 중...")
    bbbp_path = "data/bbbp/BBBP.csv"
    
    try:
        if os.path.exists(bbbp_path):
            df = pd.read_csv(bbbp_path) # p_np: 1(통과), 0(차단)
        else:
            raise FileNotFoundError(f"{bbbp_path} not found")
    except Exception as e:
        print(f">>> [Demo] {e}. 샘플 데이터로 진행합니다.")
        df = pd.DataFrame({
            'smiles': [
                'CC(=O)Oc1ccccc1C(=O)O', 
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                'CC1=CC(=C(C(=C1)C)C)C2=C(C=C(C=C2)S(=O)(=O)N)O',
                'COCC1=C(C=C(C(=C1)C#CC2=CC=CC=C2N)C)CN(C)C'
            ],
            'p_np': [1, 0, 1, 0],
            'name': ['Aspirin', 'Caffeine', 'Rofecoxib', 'EGFR-like']
        })

    # 초기화
    tree = SemanticDecisionTree()
    
    try:
        fisher = TargetFisher()
    except Exception as e:
        print(f">>> [Warning] Fisher initialization failed: {e}")
        fisher = None
        
    dto_kb = DTOKnowledgeBase("dto.rdf")
    
    print(f">>> [Process] 총 {len(df)}개 약물 분석 시작 (PubChem API 속도 고려, 상위 20개만 시연)...")
    
    # 시간 관계상 상위 20개만 테스트 (전체 분석 시 슬라이싱 제거하세요)
    # df = df.head(20) 
    test_df = df.head(20)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        smiles = row['smiles']
        is_permeable = row['p_np'] # 1 or 0
        
        # 1. Target Fishing
        if fisher:
            target_name = fisher.get_target_name(smiles)
        else:
            # Fallback if fisher failed
            target_name = row.get('name', 'Unknown')
        
        # 2. Semantic Extraction
        info = dto_kb.get_semantic_info(target_name)
        
        # 3. 데이터 추가 (통과 여부 정보도 트리에 저장)
        info['BBBP_Result'] = "Pass" if is_permeable == 1 else "Fail"
        tree.insert(info)

    print("\n>>> [Viz] 의미론적 트리 시각화 생성 중...")
    
    # 시각화 메서드 호출 (Graphviz가 설치되어 있어야 함)
    try:
        tree.export_graphviz("BBBP_Semantic_Map") 
        print(">>> [Success] 'BBBP_Semantic_Map.png' 파일이 생성되었습니다.")
    except Exception as e:
        print(f">>> [Warning] 이미지 생성 실패 (Graphviz 미설치): {e}")
        print(">>> 대신 텍스트 트리를 출력합니다.")
        tree.print_tree()

if __name__ == "__main__":
    run_bbbp_analysis()

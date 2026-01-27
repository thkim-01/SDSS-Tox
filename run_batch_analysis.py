import os
import pandas as pd
import glob
from tqdm import tqdm
from backend.semantic_decision_tree import SemanticDecisionTree, TargetFisher, DTOKnowledgeBase
import sys

# 로깅 설정 (진행률 표시를 위해 로그 레벨 조정)
import logging
logging.getLogger("SemanticTree").setLevel(logging.WARNING)

def find_columns(df):
    """SMILES와 Label 컬럼을 스마트하게 감지"""
    # SMILES 후보
    smiles_candidates = ['smiles', 'SMILES', 'mol', 'molecule', 'SMILES string', 'canonic_smiles', 'SMILES_string']
    # Label 후보
    label_candidates = ['p_np', 'activity', 'toxic', 'label', 'Class', 'outcome', 'target', 'active', 'p_np_label']
    
    found_smiles = None
    found_label = None
    
    # SMILES 찾기
    for col in df.columns:
        if any(cand.lower() == col.lower() for cand in smiles_candidates):
            found_smiles = col
            break
            
    # Label 찾기 (SMILES를 제외한 컬럼 중 선택)
    for col in df.columns:
        if col == found_smiles: continue
        if any(cand.lower() in col.lower() for cand in label_candidates):
            found_label = col
            break
            
    # 후보군에서 없으면 첫 번째 수치형 컬럼 선택 (SMILES 제외)
    if not found_label:
        for col in df.columns:
            if col == found_smiles: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                found_label = col
                break
                
    return found_smiles, found_label

def process_dataset(csv_path, fisher, dto_kb, base_results_dir):
    dataset_name = os.path.basename(os.path.dirname(csv_path))
    if not dataset_name: # 파일이 루트에 있는 경우
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        
    print(f"\n>>> [Pipeline] Analyzing Dataset: {dataset_name} ({csv_path})")
    
    results_dir = os.path.join(base_results_dir, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"!!! Error reading {csv_path}: {e}")
        return
        
    s_col, l_col = find_columns(df)
    
    if not s_col:
        print(f"!!! Could not find SMILES column in {dataset_name}. Skipping.")
        return
        
    print(f"    - Found Columns: SMILES='{s_col}', Label='{l_col or 'None'}'")
    
    tree = SemanticDecisionTree()
    
    # 상위 20개 또는 전체 분석 (사용자 요청에 따라 시연용으로 20개 제한 가능)
    # 여기서는 대규모 파이프라인이므로 전체를 돌리되, tqdm으로 진행률 표시
    # 실제 운영 시에는 속도를 위해 50~100개 정도로 제한하거나 캐시를 최대한 활용
    limit = 50 
    analysis_df = df.head(limit)
    
    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc=f"      Processing {dataset_name}"):
        smiles = str(row[s_col])
        label_val = row[l_col] if l_col else None
        
        # Target Fishing
        target_name = fisher.get_target_name(smiles)
        
        # Semantic Extraction
        info = dto_kb.get_semantic_info(target_name)
        
        # 분석 결과 추가
        if l_col:
            # 분류(Classification) vs 회귀(Regression) 판단
            if pd.api.types.is_numeric_dtype(df[l_col]):
                unique_vals = df[l_col].dropna().unique()
                if len(unique_vals) <= 5: # 분류로 간주 (0, 1 등)
                    info['BBBP_Result'] = "Pass" if label_val == 1 else "Fail"
                else: # 회귀로 간주 (용해도 등)
                    info['Value'] = f"{label_val:.2f}"
            else:
                info['Value'] = str(label_val)
                
        tree.insert(info)
        
    # 결과 저장
    output_prefix = os.path.join(results_dir, f"{dataset_name}_tree")
    try:
        tree.export_graphviz(output_prefix)
    except Exception as e:
        print(f"    - Vis Warning: {e}")
        
    # 텍스트 리포트 저장
    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print(f"Summary Report for {dataset_name}")
        print("="*40)
        tree.print_tree()
        sys.stdout = original_stdout
        
    print(f"    - Analysis complete. Results in {results_dir}")

def run_batch_analysis():
    print("="*60)
    print("       DTO-DSS Batch Analysis Pipeline")
    print("="*60)
    
    data_dir = "data"
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # 초기화
    fisher = TargetFisher(cache_path="target_cache.json")
    dto_kb = DTOKnowledgeBase("dto.rdf")
    
    # data/ 하위 모든 csv 찾기
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
                
    if not csv_files:
        print("!!! No CSV files found in data/ directory.")
        return
        
    print(f">>> Found {len(csv_files)} datasets. Starting batch analysis...")
    
    for csv_path in csv_files:
        process_dataset(csv_path, fisher, dto_kb, base_results_dir)
        
    print("\n" + "="*60)
    print("       Batch Analysis Pipeline Completed")
    print("="*60)

if __name__ == "__main__":
    run_batch_analysis()

import os
import pandas as pd
import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
from backend.app.services.predictors.semantic_decision_tree import SemanticDecisionTree, TargetFisher, DTOKnowledgeBase

#   (     )
import logging
logging.getLogger("SemanticTree").setLevel(logging.WARNING)

def find_columns(df):
    """SMILES Label   """
    # SMILES 
    smiles_candidates = ['smiles', 'SMILES', 'mol', 'molecule', 'SMILES string', 'canonic_smiles', 'SMILES_string']
    # Label 
    label_candidates = ['p_np', 'activity', 'toxic', 'label', 'Class', 'outcome', 'target', 'active', 'p_np_label']
    
    found_smiles = None
    found_label = None
    
    # SMILES 
    for col in df.columns:
        if any(cand.lower() == col.lower() for cand in smiles_candidates):
            found_smiles = col
            break
            
    # Label  (SMILES    )
    for col in df.columns:
        if col == found_smiles: continue
        if any(cand.lower() in col.lower() for cand in label_candidates):
            found_label = col
            break
            
    #        (SMILES )
    if not found_label:
        for col in df.columns:
            if col == found_smiles: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                found_label = col
                break
                
    return found_smiles, found_label

def process_dataset(csv_path, fisher, dto_kb, base_results_dir):
    dataset_name = os.path.basename(os.path.dirname(csv_path))
    if not dataset_name: #    
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
    
    #  20    (    20  )
    #     , tqdm  
    #      50~100     
    limit = 50 
    analysis_df = df.head(limit)
    
    for _, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc=f"      Processing {dataset_name}"):
        smiles = str(row[s_col])
        label_val = row[l_col] if l_col else None
        
        # Target Fishing
        target_name = fisher.get_target_name(smiles)
        
        # Semantic Extraction
        info = dto_kb.get_semantic_info(target_name)
        
        #   
        if l_col:
            # (Classification) vs (Regression) 
            if pd.api.types.is_numeric_dtype(df[l_col]):
                unique_vals = df[l_col].dropna().unique()
                if len(unique_vals) <= 5: #   (0, 1 )
                    info['BBBP_Result'] = "Pass" if label_val == 1 else "Fail"
                else: #   ( )
                    info['Value'] = f"{label_val:.2f}"
            else:
                info['Value'] = str(label_val)
                
        tree.insert(info)
        
    #  
    output_prefix = os.path.join(results_dir, f"{dataset_name}_tree")
    try:
        tree.export_graphviz(output_prefix)
    except Exception as e:
        print(f"    - Vis Warning: {e}")
        
    #   
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
    print("       SDSS-Tox Batch Analysis Pipeline")
    print("="*60)
    
    data_dir = "data"
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # 
    fisher = TargetFisher(cache_path="target_cache.json")
    dto_kb = DTOKnowledgeBase("../../data/ontology/dto.rdf")
    
    # data/   csv 
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

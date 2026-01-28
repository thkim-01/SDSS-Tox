import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.app.services.predictors.semantic_decision_tree import SemanticDecisionTree, TargetFisher, DTOKnowledgeBase
from tqdm import tqdm

def run_bbbp_analysis():
    print(">>> [System] BBBP   ...")
    bbbp_path = "data/bbbp/BBBP.csv"
    
    try:
        if os.path.exists(bbbp_path):
            df = pd.read_csv(bbbp_path) # p_np: 1(), 0()
        else:
            raise FileNotFoundError(f"{bbbp_path} not found")
    except Exception as e:
        print(f">>> [Demo] {e}.   .")
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

    # 
    tree = SemanticDecisionTree()
    
    try:
        fisher = TargetFisher()
    except Exception as e:
        print(f">>> [Warning] Fisher initialization failed: {e}")
        fisher = None
        
    dto_kb = DTOKnowledgeBase("../../data/ontology/dto.rdf")
    
    print(f">>> [Process]  {len(df)}    (PubChem API  ,  20 )...")
    
    #    20  (    )
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
        
        # 3.   (    )
        info['BBBP_Result'] = "Pass" if is_permeable == 1 else "Fail"
        tree.insert(info)

    print("\n>>> [Viz]     ...")
    
    #    (Graphviz   )
    try:
        tree.export_graphviz("BBBP_Semantic_Map") 
        print(">>> [Success] 'BBBP_Semantic_Map.png'  .")
    except Exception as e:
        print(f">>> [Warning]    (Graphviz ): {e}")
        print(">>>    .")
        tree.print_tree()

if __name__ == "__main__":
    run_bbbp_analysis()

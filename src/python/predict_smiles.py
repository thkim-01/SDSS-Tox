"""
Standalone SMILES prediction script for JavaFX GUI integration.
Reads SMILES from a file and outputs predictions.
"""
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from model import MoleculeModel
from data_loader import DataLoader

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_smiles.py <smiles_file>")
        sys.exit(1)
    
    smiles_file = sys.argv[1]
    
    # Read SMILES from file
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    if not smiles_list:
        print("ERROR: No SMILES provided")
        sys.exit(1)
    
    # Load data and train model
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    loader = DataLoader(data_path)
    
    try:
        X, y, tasks = loader.get_data('tox21')
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)
    
    # Train model
    model = MoleculeModel(n_estimators=50)
    model.train(X, y, task_name='NR-AhR')
    
    # Predict for each SMILES
    for smiles in smiles_list:
        try:
            pred = model.predict(smiles)
            if pred is not None:
                prob = pred[0]
                risk = "High" if prob > 0.5 else "Low"
                print(f"RESULT:{smiles}|{prob:.4f}|{risk}")
            else:
                print(f"RESULT:{smiles}|ERROR|Invalid")
        except Exception as e:
            print(f"RESULT:{smiles}|ERROR|{str(e)}")

if __name__ == '__main__':
    main()

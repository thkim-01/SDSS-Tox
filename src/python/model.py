import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Warning: RDKit not found. Featurization will fail.")
    Chem = None

class MoleculeModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.tasks = []

    def featurize(self, smiles_list):
        if Chem is None:
            raise ImportError("RDKit is required for featurization.")
            
        fps = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fps.append(np.array(fp))
                    valid_indices.append(i)
            except Exception as e:
                # Ignore invalid SMILES
                continue
        return np.array(fps), valid_indices

    def train(self, X_smiles, y_df, task_name='NR-AhR'):
        self.tasks = y_df.columns.tolist()
        print("Featurizing molecules...")
        X_fps, valid_indices = self.featurize(X_smiles)
        y_filtered = y_df.iloc[valid_indices]

        if task_name not in y_filtered.columns:
            raise ValueError(f"Task {task_name} not found in dataset")

        print(f"Training on task: {task_name}")
        
        y_task = y_filtered[task_name]
        mask = ~y_task.isna()
        X_final = X_fps[mask]
        y_final = y_task[mask]

        if len(X_final) == 0:
            print("No valid data for this task.")
            return 0.0

        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        print(f"ROC-AUC for {task_name}: {score:.4f}")
        
        return score

    def predict(self, smiles):
        # Single molecule prediction
        X_fps, _ = self.featurize([smiles])
        if len(X_fps) == 0:
            return None
        return self.model.predict_proba(X_fps)[:, 1]

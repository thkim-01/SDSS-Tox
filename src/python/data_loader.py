import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_tox21(self):
        # Handle both absolute and relative paths
        if os.path.isabs(self.data_dir):
            file_path = os.path.join(self.data_dir, 'tox21', 'tox21.csv')
        else:
             file_path = os.path.join(os.getcwd(), self.data_dir, 'tox21', 'tox21.csv')
             
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df

    def get_data(self, dataset_name='tox21'):
        if dataset_name == 'tox21':
            df = self.load_tox21()
            # Tasks are the first 12 columns
            tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
                     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
            
            # Filter out rows with missing SMILES
            df = df.dropna(subset=['smiles'])
            
            return df['smiles'].values, df[tasks], tasks
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

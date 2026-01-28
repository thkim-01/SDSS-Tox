"""
Dataset Loader - MoleculeNet   
- data/  CSV  
- SMILES   
-  
"""

import os
import csv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not found. Descriptor calculation will be skipped.")


@dataclass
class DatasetInfo:
    """ """
    name: str
    path: str
    size: int
    smiles_column: str
    label_columns: List[str]
    description: str
    type: str = "classification"


@dataclass
class DatasetSample:
    """ """
    smiles: str
    labels: Dict[str, Optional[float]]
    descriptors: Dict[str, float] = None
    name: str = "Unknown"


class DatasetLoader:
    """MoleculeNet   ( )"""
    
    #    -    
    # esol, freesolv, lipophilicity, muv, qm7, qm8 
    DATASET_CONFIG = {
        'tox21': {
            'smiles_column': 'smiles',
            'description': 'Tox21   (12 )',
            'label_columns': [
                'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                'SR-HSE', 'SR-MMP', 'SR-p53'
            ]
        },
        'bbbp': {
            'smiles_column': 'smiles',
            'description': 'Blood-Brain Barrier Permeability',
            'label_columns': ['p_np']
        },
        'clintox': {
            'smiles_column': 'smiles',
            'description': '  ',
            'label_columns': ['FDA_APPROVED', 'CT_TOX']
        },
        'hiv': {
            'smiles_column': 'smiles',
            'description': 'HIV  ',
            'label_columns': ['HIV_active']
        },
        'sider': {
            'smiles_column': 'smiles',
            'description': '  (27 )',
            'label_columns': []  #  
        },
        'bace': {
            'smiles_column': 'mol',
            'description': 'BACE-1  ',
            'label_columns': ['Class'],
            'type': 'classification'
        }
        #  : esol, freesolv, lipophilicity, muv, qm7, qm8, qm9
    }
    
    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir:   
        """
        self.data_dir = data_dir or self._find_data_dir()
        self.available_datasets = self._scan_datasets()
        
        count = len(self.available_datasets)
        logger.info(f"DatasetLoader initialized. Found {count} datasets")
        
    def _find_data_dir(self) -> str:
        """  """
        possible_paths = [
            "data",
            "../data",
            "../../data",
            os.path.join(os.path.dirname(__file__), "../../../data"),
        ]
        
        for path in possible_paths:
            if os.path.isdir(path):
                abs_path = os.path.abspath(path)
                logger.info(f"Found data directory at: {abs_path}")
                return abs_path
                
        msg = "Data directory not found in standard locations. "
        msg += "Defaulting to 'data'"
        logger.warning(msg)
        return "data"
        
    def _scan_datasets(self) -> Dict[str, DatasetInfo]:
        """    ( )"""
        datasets = {}
        
        #   
        excluded = {
            'esol', 'freesolv', 'lipophilicity', 'muv',
            'qm7', 'qm8', 'qm9'
        }
        
        if not os.path.isdir(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return datasets
            
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            dataset_name_lower = item.lower()
            
            #    
            if dataset_name_lower in excluded:
                msg = f"Skipping excluded dataset: {dataset_name_lower}"
                logger.info(msg)
                continue
            
            if os.path.isdir(item_path):
                #  CSV 
                for file in os.listdir(item_path):
                    if file.endswith('.csv'):
                        csv_path = os.path.join(item_path, file)
                        dataset_name = item.lower()
                        
                        info = self._get_dataset_info(dataset_name, csv_path)
                        if info:
                            datasets[dataset_name] = info
                            
            elif item.endswith('.csv'):
                #  CSV 
                dataset_name = os.path.splitext(item)[0].lower()
                
                #   
                if dataset_name not in excluded:
                    info = self._get_dataset_info(dataset_name, item_path)
                    if info:
                        datasets[dataset_name] = info
                    
        return datasets
        
    def _get_dataset_info(self, name: str, path: str) -> Optional[DatasetInfo]:
        """  """
        try:
            # CSV  
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                #   
                row_count = sum(1 for _ in reader)
                
            # SMILES  
            smiles_col = None
            for col in ['smiles', 'SMILES', 'Smiles', 'mol', 'molecule']:
                if col in header:
                    smiles_col = col
                    break
                    
            if not smiles_col:
                logger.warning(f"No SMILES column found in {path}")
                return None
                
            #   (   )
            config = self.DATASET_CONFIG.get(name, {})
            label_cols = config.get('label_columns', [])
            
            if not label_cols:
                # SMILES     
                label_cols = [c for c in header if c != smiles_col]
                
            return DatasetInfo(
                name=name,
                path=path,
                size=row_count,
                smiles_column=smiles_col,
                label_columns=label_cols,
                description=config.get('description', f'{name} dataset'),
                type=config.get('type', 'classification') # Default
            )
            
        except Exception as e:
            logger.error(f"Error reading dataset {path}: {e}")
            return None
            
    def list_datasets(self) -> List[Dict]:
        """    """
        return [
            {
                'name': info.name,
                'size': info.size,
                'smiles_column': info.smiles_column,
                'label_columns': info.label_columns[:5],  #  5
                'label_columns': info.label_columns[:5],  #  5
                'description': info.description,
                'type': info.type
            }
            for info in self.available_datasets.values()
        ]
        
    
    def load_all(self) -> List[Dict]:
        """Load all samples from all available datasets.
        Returns a list of dicts with keys:
            - "smiles": SMILES string
            - "descriptors": dict of descriptor values (all columns except SMILES and label columns)
        """
        all_samples: List[Dict] = []
        for name, info in self.available_datasets.items():
            try:
                with open(info.path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        smiles = row.get(info.smiles_column, '').strip()
                        if not smiles:
                            continue
                        # descriptors: all columns except SMILES and label columns
                        descriptor_dict = {}
                        for col, val in row.items():
                            if col == info.smiles_column or col in info.label_columns:
                                continue
                            if val == '' or val is None:
                                descriptor_dict[col] = None
                            else:
                                try:
                                    descriptor_dict[col] = float(val)
                                except ValueError:
                                    descriptor_dict[col] = val
                        all_samples.append({
                            "smiles": smiles,
                            "descriptors": descriptor_dict,
                        })
            except Exception as e:
                logger.error(f"Error loading dataset {name}: {e}")
        logger.info(f"Loaded total {len(all_samples)} samples across all datasets")
        return all_samples


    def load_dataset(
        self, 
        name: str, 
        sample_size: Optional[int] = None,
        label_column: Optional[str] = None
    ) -> Tuple[List[DatasetSample], DatasetInfo]:
        """
         
        
        Args:
            name:  
            sample_size:   (None = )
            label_column:    (None =  )
            
        Returns:
            (samples, dataset_info)
        """
        name = name.lower()
        
        if name not in self.available_datasets:
            raise ValueError(f"Dataset not found: {name}. Available: {list(self.available_datasets.keys())}")
            
        info = self.available_datasets[name]
        samples = []
        
        try:
            with open(info.path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if sample_size and i >= sample_size:
                        break
                        
                    smiles = row.get(info.smiles_column, '').strip()
                    if not smiles:
                        continue
                        
                    #  
                    labels = {}
                    for col in info.label_columns:
                        val = row.get(col, '').strip()
                        if val:
                            try:
                                labels[col] = float(val)
                            except ValueError:
                                labels[col] = None
                        else:
                            labels[col] = None
                            
                    # (Descriptors) : SMILES     
                    descriptors = {}
                    for col, val in row.items():
                        if col == info.smiles_column or col in info.label_columns:
                            continue
                        try:
                            if val:
                                descriptors[col] = float(val)
                        except (ValueError, TypeError):
                            continue #   
                            
                    # If descriptors are missing or suspiciously empty (LogP=0, MW=0), try to compute
                    # Many benchmark datasets only have SMILES and Labels
                    if RDKIT_AVAILABLE and (not descriptors or descriptors.get("MW", 0) == 0 or descriptors.get("LogP", 0) == 0):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                descriptors["MW"] = Descriptors.MolWt(mol)
                                descriptors["logP"] = Crippen.MolLogP(mol) # Explicitly use Crippen for LogP
                                descriptors["logKow"] = descriptors["logP"]
                                descriptors["HBD"] = Lipinski.NumHDonors(mol)
                                descriptors["HBA"] = Lipinski.NumHAcceptors(mol)
                                descriptors["nRotB"] = Lipinski.NumRotatableBonds(mol)
                                descriptors["TPSA"] = Descriptors.TPSA(mol)
                                descriptors["Aromatic_Rings"] = Lipinski.NumAromaticRings(mol)
                                descriptors["Heteroatom_Count"] = Lipinski.NumHeteroatoms(mol)
                                descriptors["Heavy_Atom_Count"] = mol.GetNumHeavyAtoms()
                        except Exception as e:
                            # If calc fails, skip sample or keep empty?
                            # For visualization, we need descriptors. Better to skip if critical ones miss.
                            logger.debug(f"Failed to re-compute descriptors for {smiles}: {e}")

                    # Validation: Filter out invalid samples (Infinity or NaN)
                    is_valid = True
                    for k, v in descriptors.items():
                        if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                            is_valid = False
                            break
                    
                    if not is_valid:
                        continue

                    samples.append(DatasetSample(
                        smiles=smiles,
                        labels=labels,
                        descriptors=descriptors,
                        name=row.get("name") or row.get("Name") or row.get("Compound ID") or "Unknown"
                    ))
                    
            logger.info(f"Loaded {len(samples)} samples from {name}")
            return samples, info
            
        except Exception as e:
            logger.error(f"Error loading dataset {name}: {e}")
            raise
            
    def get_smiles_list(
        self, 
        name: str, 
        sample_size: Optional[int] = None
    ) -> List[str]:
        """SMILES  """
        samples, _ = self.load_dataset(name, sample_size)
        return [s.smiles for s in samples]
        
        
#  
_dataset_loader = None

def get_dataset_loader() -> DatasetLoader:
    """DatasetLoader   """
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader()
    return _dataset_loader

import argparse
from data_loader import DataLoader
from model import MoleculeModel
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='DTO-DSS Python Backend')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Path to data directory')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--task', type=str, default='NR-AhR', help='Task to train on')
    args = parser.parse_args()

    # Resolve data path
    # If running from src/python, default ../../data is correct
    # If running from root, data is correct
    
    if os.path.exists(args.data_dir):
        data_path = args.data_dir
    elif os.path.exists(os.path.join(os.path.dirname(__file__), args.data_dir)):
        data_path = os.path.join(os.path.dirname(__file__), args.data_dir)
    else:
        # Fallback for absolute path construction if needed
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

    print(f"Loading data from {data_path}")
    try:
        loader = DataLoader(data_path)
        X, y, tasks = loader.get_data('tox21')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if args.train:
        model = MoleculeModel()
        try:
            model.train(X, y, task_name=args.task)
        except Exception as e:
            print(f"Error during training: {e}")

if __name__ == '__main__':
    main()

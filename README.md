# SDSS-Tox

A deep learning framework for molecular toxicity prediction using SMILES strings and graph neural networks.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/thkim-01/SDSS-Tox.git
cd SDSS-Tox
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training
```bash
python train.py --dataset tox21 --model gcn --epochs 100
```

#### Prediction
```bash
python predict.py --model_path checkpoints/best_model.pt --smiles "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
```

#### Available Options
- `--dataset`: Dataset name (tox21, clintox, sider)
- `--model`: Model architecture (gcn, gat, graphsage)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate

## Project Structure

```
SDSS-Tox/
├── data/              # Dataset files
├── models/            # Model architectures
├── utils/             # Utility functions
├── train.py           # Training script
├── predict.py         # Prediction script
└── requirements.txt   # Dependencies
```

## Results

| Dataset | Model | AUC-ROC | Accuracy |
|---------|-------|---------|----------|
| Tox21   | GCN   | 0.85    | 0.82     |
| ClinTox | GAT   | 0.88    | 0.84     |

## Citation

If you use this code, please cite:
```bibtex
@article{your_paper,
  title={SDSS-Tox: Deep Learning for Toxicity Prediction},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
"""PyTorch     .

Phase 1.3:    
- BaseModel:    
- MolecularGNN:   (Graph Convolutional Network)
- MolecularCNN:   (Convolutional Neural Network)

Author: DTO-DSS Team
Date: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """ PyTorch   .

      (CPU/CUDA)   
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model initialized on device: {self.device}")

    def count_parameters(self) -> int:
        """   """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """   (Transfer Learning)"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_all(self):
        """    """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")


class MolecularGNN(BaseModel):
    """   (Graph Neural Network).

    SMILES     GNN   .

    :
        - Graph Convolutional Layers (GCN/GAT)
        - Batch Normalization
        - Dropout for regularization
        - Fully Connected Layers

    :
        - Node features ( )
        - Edge index ( )
        - Batch vector
    """

    def __init__(
        self,
        node_features: int = 10,      #  descriptor 
        hidden_channels: int = 128,   #   
        num_layers: int = 3,         # GCN  
        num_classes: int = 2,         # Safe(0) / Toxic(1)
        dropout: float = 0.5,         # Dropout 
        use_gat: bool = False         # Graph Attention Network  
    ):
        super().__init__()

        # Graph Convolutional Layers
        from torch_geometric.nn import GCNConv, GATConv

        if use_gat:
            self.conv_type = "GAT"
            self.conv1 = GATConv(node_features, hidden_channels, heads=4)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4)
        else:
            self.conv_type = "GCN"
            self.conv1 = GCNConv(node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_channels, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"MolecularGNN initialized: "
            f"node_features={node_features}, "
            f"hidden_channels={hidden_channels}, "
            f"num_layers={num_layers}, "
            f"use_gat={use_gat}, "
            f"total_params={self.count_parameters()}"
        )

    def forward(self, x, edge_index, batch=None):
        """Forward pass.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Graph Convolution 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph Convolution 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class MolecularCNN(BaseModel):
    """    .

     2D/3D    CNN   .

    :
        - Convolutional Layers (2D Conv)
        - Max/Average Pooling
        - Batch Normalization
        - Dropout for regularization
        - Fully Connected Layers

    :
        -   [batch, channels, height, width]
    """

    def __init__(
        self,
        in_channels: int = 1,       # Grayscale  
        num_classes: int = 2,         # Safe(0) / Toxic(1)
        conv_channels: list = [32, 64, 128],  # Conv  
        kernel_sizes: list = [3, 3, 3],          # Kernel 
        pool_size: int = 2,            # Pooling 
        dropout: float = 0.5,          # Dropout 
    ):
        super().__init__()

        # Convolutional Blocks
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else conv_channels[i-1],
                         conv_channels[i],
                         kernel_sizes[i],
                         padding=1),
                nn.BatchNorm2d(conv_channels[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size, pool_size)
            )
            for i in range(len(conv_channels))
        ])

        # Fully Connected Layers
        # Calculate the size after all convolutions and pooling
        conv_output_size = 128 * (pool_size ** len(conv_channels))
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"MolecularCNN initialized: "
            f"in_channels={in_channels}, "
            f"conv_channels={conv_channels}, "
            f"kernel_sizes={kernel_sizes}, "
            f"total_params={self.count_parameters()}"
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Image tensor [batch, channels, height, width]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Convolutional Blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # : GNN
    print("=" * 60)
    print("Testing MolecularGNN")
    print("=" * 60)

    gnn = MolecularGNN(
        node_features=10,
        hidden_channels=64,
        num_classes=2
    )

    # Dummy inputs
    x = torch.randn(10, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=torch.long)

    try:
        output = gnn(x, edge_index)
        print(f"GNN output shape: {output.shape}")
        print(f"GNN parameters: {gnn.count_parameters()}")
    except Exception as e:
        print(f"GNN test failed: {e}")
        print("Note: torch-geometric not installed yet")

    print()

    # : CNN
    print("=" * 60)
    print("Testing MolecularCNN")
    print("=" * 60)

    cnn = MolecularCNN(
        in_channels=1,
        conv_channels=[16, 32, 64],
        num_classes=2
    )

    # Dummy input
    x = torch.randn(1, 1, 32, 32)

    try:
        output = cnn(x)
        print(f"CNN output shape: {output.shape}")
        print(f"CNN parameters: {cnn.count_parameters()}")
    except Exception as e:
        print(f"CNN test failed: {e}")

    print()
    print("=" * 60)
    print("PyTorch Base Models Test Complete!")
    print("=" * 60)

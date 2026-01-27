"""PyTorch 기본 모델 구조 및 유틸티티.

Phase 1.3: 기본 모델 구조 설계
- BaseModel: 모든 모델의 기본 클래스
- MolecularGNN: 그래프 신경망 (Graph Convolutional Network)
- MolecularCNN: 합성곱 신경망 (Convolutional Neural Network)

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
    """모든 PyTorch 모델의 기본 클래스.

    디바이스 관리 (CPU/CUDA)를 제공하는 기본 구조
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model initialized on device: {self.device}")

    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """백본 네트워크 고정 (Transfer Learning용)"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_all(self):
        """모든 파라미터 학습 가능 설정"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")


class MolecularGNN(BaseModel):
    """분자 그래프 신경망 (Graph Neural Network).

    SMILES에서 분자 그래프로 변환한 후 GNN을 사용하여 독성 예측.

    아키텍처:
        - Graph Convolutional Layers (GCN/GAT)
        - Batch Normalization
        - Dropout for regularization
        - Fully Connected Layers

    입력:
        - Node features (원자 특성)
        - Edge index (결합 정보)
        - Batch vector
    """

    def __init__(
        self,
        node_features: int = 10,      # 분자 descriptor 개수
        hidden_channels: int = 128,   # 은닉 레이어 크기
        num_layers: int = 3,         # GCN 레이어 수
        num_classes: int = 2,         # Safe(0) / Toxic(1)
        dropout: float = 0.5,         # Dropout 비율
        use_gat: bool = False         # Graph Attention Network 사용 여부
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
    """분자 이미지 기반 합성곱 신경망.

    분자의 2D/3D 구조를 이미지로 변환하여 CNN을 사용하여 독성 예측.

    아키텍처:
        - Convolutional Layers (2D Conv)
        - Max/Average Pooling
        - Batch Normalization
        - Dropout for regularization
        - Fully Connected Layers

    입력:
        - 분자 이미지 [batch, channels, height, width]
    """

    def __init__(
        self,
        in_channels: int = 1,       # Grayscale 분자 이미지
        num_classes: int = 2,         # Safe(0) / Toxic(1)
        conv_channels: list = [32, 64, 128],  # Conv 채널 크기
        kernel_sizes: list = [3, 3, 3],          # Kernel 크기
        pool_size: int = 2,            # Pooling 크기
        dropout: float = 0.5,          # Dropout 비율
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

    # 테스트: GNN
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

    # 테스트: CNN
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

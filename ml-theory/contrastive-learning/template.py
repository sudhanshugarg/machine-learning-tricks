"""
Template for Contrastive Learning Experiments

This template provides a starter structure for implementing and experimenting
with contrastive learning objectives (InfoNCE, NT-Xent, triplet loss).

Follow the TODOs and run the script to see concrete behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Simple CNN encoder for demonstration.
    In practice, replace with ResNet, ViT, or domain-specific architecture.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ProjectionHead(nn.Module):
    """
    MLP projection head as used in SimCLR.
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """
    Combined encoder + projection head for contrastive learning.
    """

    def __init__(self, encoder: nn.Module = None, projection: nn.Module = None):
        super().__init__()
        self.encoder = encoder or Encoder()
        self.projection = projection or ProjectionHead()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (representation h, projection z).
        Use h for downstream tasks; z for contrastive loss.
        """
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


class ContrastiveLosses:
    """
    Collection of contrastive loss functions.
    """

    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy (SimCLR).

        Args:
            z1, z2: [B, D] projected embeddings from two augmentations.
            temperature: softmax temperature.

        Returns:
            Scalar loss.
        """
        # TODO: Implement NT-Xent loss
        # 1. Normalize z1 and z2
        # 2. Concatenate to [2B, D]
        # 3. Compute similarity matrix [2B, 2B]
        # 4. Create positive mask
        # 5. Compute cross-entropy
        raise NotImplementedError("See code_examples.py for reference implementation.")

    @staticmethod
    def info_nce(
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        InfoNCE with explicit negatives.

        Args:
            query: [B, D] anchor embeddings.
            positive: [B, D] positive embeddings.
            negatives: [N, D] negative embeddings.
            temperature: softmax temperature.

        Returns:
            Scalar loss.
        """
        # TODO: Implement InfoNCE
        raise NotImplementedError("See code_examples.py for reference implementation.")

    @staticmethod
    def triplet_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Standard triplet loss.

        Args:
            anchor, positive, negative: [B, D] embeddings.
            margin: hinge margin.

        Returns:
            Scalar loss.
        """
        # TODO: Implement triplet loss
        raise NotImplementedError("See code_examples.py for reference implementation.")


class ContrastiveTrainer:
    """
    Interactive demonstration of contrastive training.
    """

    def __init__(self, model: ContrastiveModel = None, temperature: float = 0.5):
        self.model = model or ContrastiveModel()
        self.temperature = temperature
        self.criterion = ContrastiveLosses()

    def train_step(self, x1: torch.Tensor, x2: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Single training step with NT-Xent."""
        self.model.train()
        optimizer.zero_grad()

        _, z1 = self.model(x1)
        _, z2 = self.model(x2)

        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        loss = self.criterion.nt_xent(z1, z2, self.temperature)
        loss.backward()
        optimizer.step()
        return loss.item()

    def extract_representations(self, dataloader) -> torch.Tensor:
        """Extract h (not z) representations for evaluation."""
        self.model.eval()
        reps = []
        with torch.no_grad():
            for x, _ in dataloader:
                h, _ = self.model(x)
                reps.append(h)
        return torch.cat(reps, dim=0)


def main():
    """
    TODO: Run these experiments and observe the output.
    """
    # Create dummy data mimicking two augmented views
    batch_size = 8
    x1 = torch.randn(batch_size, 3, 32, 32)
    x2 = torch.randn(batch_size, 3, 32, 32)

    model = ContrastiveModel()
    trainer = ContrastiveTrainer(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 1. Run a training step
    print("Running training step...")
    # loss = trainer.train_step(x1, x2, optimizer)
    # print(f"Loss: {loss:.4f}")

    # 2. Extract representations
    print("\nExtracting representations...")
    # reps = trainer.extract_representations([(x1, None)])
    # print(f"Representation shape: {reps.shape}")

    # 3. Inspect model architecture
    print("\nModel architecture:")
    print(model)


if __name__ == "__main__":
    main()

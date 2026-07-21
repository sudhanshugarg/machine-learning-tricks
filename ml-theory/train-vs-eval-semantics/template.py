"""
Template for Train vs Eval Semantics

This template provides a starter structure for experimenting with
PyTorch train/eval mode differences, writing tests, and inspecting
BatchNorm running statistics.

Follow the TODOs and run the script to see concrete behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoModel(nn.Module):
    """
    A small model containing Dropout and BatchNorm to demonstrate
    train vs eval semantics.
    """

    def __init__(self, dropout_p: float = 0.5, num_features: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.bn = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop(x)
        return self.fc2(x)


class TrainEvalDemo:
    """
    Interactive demonstration of train/eval semantics.
    """

    def __init__(self, model: nn.Module = None):
        self.model = model or DemoModel()

    def show_mode_state(self):
        """Print training flag for every submodule."""
        for name, module in self.model.named_modules():
            print(f"{name:20s} training={module.training}")

    def compare_outputs(self, x: torch.Tensor):
        """Run the same input twice in each mode and print equality."""
        print("\n--- TRAIN MODE ---")
        self.model.train()
        out1 = self.model(x)
        out2 = self.model(x)
        print(f"out1 close to out2? {torch.allclose(out1, out2, atol=1e-6)}")
        print(f"Max diff: {(out1 - out2).abs().max().item():.6f}")

        print("\n--- EVAL MODE ---")
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        print(f"out1 close to out2? {torch.allclose(out1, out2, atol=1e-6)}")
        print(f"Max diff: {(out1 - out2).abs().max().item():.6f}")

    def inspect_bn_stats(self):
        """Display current BatchNorm running stats and num_batches_tracked."""
        bn = self.model.bn
        print("\n--- BatchNorm Stats ---")
        print(f"running_mean:          {bn.running_mean[:5]}")  # first 5
        print(f"running_var:           {bn.running_var[:5]}")
        print(f"num_batches_tracked:   {bn.num_batches_tracked.item()}")

    def train_step(self, x: torch.Tensor, target: torch.Tensor):
        """Single training step (forward + backward)."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        pred = self.model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single evaluation step (no gradients, eval mode)."""
        self.model.eval()
        with torch.inference_mode():
            return self.model(x)


def main():
    """
    TODO: Run these experiments and observe the output.
    """
    demo = TrainEvalDemo()
    x = torch.randn(8, 10)
    target = torch.randn(8, 1)

    # 1. Inspect mode flags
    print("Initial mode state:")
    demo.show_mode_state()

    # 2. See how outputs differ between train and eval
    demo.compare_outputs(x)

    # 3. Inspect BN stats before training
    demo.inspect_bn_stats()

    # 4. Train for a few steps and watch BN stats change
    print("\n--- Training 3 steps ---")
    for i in range(3):
        loss = demo.train_step(x, target)
        print(f"Step {i+1} loss: {loss:.4f}")
    demo.inspect_bn_stats()

    # 5. Run eval and confirm BN stats freeze
    print("\n--- Eval step (stats should NOT change) ---")
    _ = demo.eval_step(x)
    demo.inspect_bn_stats()


if __name__ == "__main__":
    main()

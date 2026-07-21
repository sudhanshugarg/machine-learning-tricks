# Debugging a Model That Won't Learn

## The Scenario

You are training a simple CNN on CIFAR-10. After 10 epochs, your training loss is stuck around 2.3 (random-guess level for 10 classes) and accuracy is ~10%. Something is clearly broken, but the code runs without crashing.

Your task:
1. **Diagnose** what is preventing the model from learning.
2. **Explain** the root cause of each bug using ML theory (gradients, activation dynamics, optimization, etc.).
3. **Fix** the code and verify the model learns correctly.

You are allowed to run the code, inspect tensors, print gradients, and add debugging instrumentation. Treat this as a real debugging session.

---

## The Buggy Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BrokenCNN(nn.Module):
    """A CNN that should classify CIFAR-10 but doesn't learn."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

        # Bug source 1: weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, val=0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    # Bug source 2: data loading
    transform = transforms.Compose([
        transforms.ToTensor(),  # maps [0,255] -> [0,1]
        # No normalization!
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = BrokenCNN()
    # Bug source 3: loss function mismatch
    criterion = nn.MSELoss()

    # Bug source 4: optimizer settings
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0, momentum=0.9)

    # Bug source 5: training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            # Bug source 6: device mismatch (if CUDA available)
            images, labels = images.cuda(), labels.cuda()
            model.cuda()

            outputs = model(images)

            # Bug source 7: target shape / type mismatch for MSELoss
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()
            # Bug source 8: missing gradient zeroing

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

    # Quick eval
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train()
```

---

## Your Task

1. **Run the code** (or mentally trace it). What symptoms do you observe?
2. **Systematically debug**. For each suspected issue:
   - What is the theoretical reason it breaks learning?
   - What empirical evidence (prints, gradient norms, activation histograms) would confirm it?
   - What is the fix?
3. **Produce a corrected version** that reaches >60% test accuracy on CIFAR-10 within 10 epochs.

---

## Open-Ended Discussion Questions

After you fix the code, consider these:

1. **Vanishing vs. Exploding Gradients**: If the model had very deep layers (e.g., 50-layer ResNet) instead of this shallow CNN, what *additional* failure modes would you watch for? How does BatchNorm theoretically prevent them?

2. **Dead ReLUs**: Even after fixing the zero-initialization, could ReLU neurons still "die" during training? What does "dead" mean theoretically (gradient = 0 forever), and what mitigations exist?

3. **Learning Rate Diagnostics**: The buggy code uses `lr=10.0`. If you had to *blindly* tune the learning rate without looking at loss curves, what theoretical heuristics exist (e.g., LR range test, largest stable LR)?

4. **Data Normalization**: CIFAR-10 pixel values are in `[0, 1]`. Why does subtracting the per-channel mean (e.g., `Normalize((0.4914, 0.4822, 0.4465), ...)`) help optimization theoretically? (Hint: consider the condition number of the Hessian and gradient variance.)

5. **When Loss Goes NaN**: If the loss suddenly becomes `NaN` mid-training, what is the most likely theoretical cause? How would you confirm it, and what are three different architectural fixes?

---

## Deliverables

- A list of **all bugs** you found, ranked by severity.
- A **brief theoretical explanation** for why each bug kills learning.
- **Empirical evidence** you used to confirm each bug.
- The **fully corrected code** with comments marking each fix.

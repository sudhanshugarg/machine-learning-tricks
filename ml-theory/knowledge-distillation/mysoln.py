"""
given a teacher model
create a student model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SingleFFN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.Dropout(0.2))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)

class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int):
        super().__init__()

        self.fc1 = SingleFFN(input_dim, hidden_layer_dim)
        self.fc2 = SingleFFN(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = SingleFFN(hidden_layer_dim, hidden_layer_dim)
        self.fc4 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x: torch.Tensor):
        #x is b, input_dim
        self.o1 = self.fc1(x) #b, h_dim
        self.o2 = self.o1 + self.fc2(self.o1) #b, h_dim
        self.o3 = self.o2 + self.fc3(self.o2) #b, h_dim
        return self.fc4(self.o3) #b, o_dim

class DistillationLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.2,
                 beta: float = 0.8,
                 temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor):
        normal_cross_entropy_loss = F.cross_entropy(student_logits, hard_labels)
        # student_logits: b
        student_log_softmax = F.log_softmax(student_logits / self.T, dim=1)
        teacher_softmax = F.softmax(teacher_logits / self.T, dim=1)

        kl_div_loss = (self.T ** 2) * F.kl_div(student_log_softmax, teacher_softmax, reduction="batchmean")
        loss = self.alpha * normal_cross_entropy_loss + self.beta * kl_div_loss
        return loss
        

class ImageData(Dataset):
    def __init__(self, dim: int = 768):
        self.n = 10000
        self.x = torch.randint(low=0, high=256, size=(self.n, dim))
        self.y = torch.randint(low=0, high=10, size=(self.n,))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.n


def warmup(step):
    if step < 1000:
        return step / 1000.0
    return 1.0

def train():
    teacher = FFN(input_dim=768, hidden_layer_dim=1536, output_dim=10)
    student = FFN(input_dim=768, hidden_layer_dim=128, output_dim=10)
    lossFn = DistillationLoss()

    dataset = ImageData()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    epochs = 100
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, betas=(0.95, 0.995), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    student.train()
    teacher.eval()

    for i in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)

            loss = lossFn(student_logits, teacher_logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"epoch {i}, avg_loss = {avg_loss:.2f}")

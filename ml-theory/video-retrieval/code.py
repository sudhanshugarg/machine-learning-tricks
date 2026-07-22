"""
We want to build a real-time content recommendation
system that suggests short form videos to users. However, we have two
competing goals: we want to maximize the probability that a user clicks and watches a video to completion, but we also want to minimize the
probability that they report or dislike the video. Walk us through how you would design this end to end, and write the PyTorch code for the
model architecture and the custom loss function.
"""


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class TwoTowerModel(nn.Module):
    MAX_USERS = 10000
    MAX_VIDEOS = 1000000
    def __init__(self, device=None):
        super().__init__()
        self.user_embeddings = nn.Embedding(self.MAX_USERS, 32, device=device)
        self.video_embeddings = nn.Embedding(self.MAX_VIDEOS, 32, device=device)

        self.user_tower = nn.Sequential(
            nn.Linear(32, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32)
        )

        self.video_tower = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 32)
        )


    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids, video_ids = x # each is [batch]
        user_embeddings = self.user_embeddings(user_ids) #b, 32
        video_embeddings = self.video_embeddings(video_ids) #b, 32

        user_logits = self.user_tower(user_embeddings)
        video_logits = self.video_tower(video_embeddings)

        return user_logits, video_logits


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.t = temperature
    

    def stable_cross_entropy_loss(self, logits: torch.Tensor):
        batch = logits.shape[0]
        labels = logits[torch.arange(batch), torch.arange(batch)][:, None]
        # print("labels: ", labels)
        max_logits = torch.max(logits, dim=1, keepdim=True).values
        # print("max_logits: ", max_logits)
        # print("softmax/max: ", softmax / max_logits)
        # print("sum(softmax/max: )", torch.sum(softmax / max_logits, dim=1, keepdim=True))
        # print("log(sum(softmax/max: ))", torch.log(torch.sum(softmax / max_logits, dim=1, keepdim=True)))
        log_loss = labels - max_logits - torch.log(torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True))
        # print("way1: ", torch.mean(-log_loss).item())
        softmax = F.softmax(logits, dim=1)
        way2 = torch.mean(-torch.log(softmax[torch.arange(batch), torch.arange(batch)]))
        # print("way2: ", way2.item())
        return torch.mean(-log_loss)


    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        assert logits1.shape == logits2.shape

        batch, embedding_dim = logits1.shape
        #create a matrix of batch x batch
        #each cell is a dot product between softmax values

        dot_product = torch.matmul(logits1, logits2.T) #batch, batch
        # print(dot_product.shape)
        
        labels = torch.arange(batch)
        loss1 = F.cross_entropy(dot_product, labels)
        # loss1_manual = self.stable_cross_entropy_loss(dot_product)
        # print("loss1_manual: ", loss1_manual.item(), "loss1: ", loss1.item())
        loss2 = F.cross_entropy(dot_product.T, labels)
        # loss2_manual = self.stable_cross_entropy_loss(dot_product.T)
        # print("loss2_manual: ", loss2_manual.item(), "loss2: ", loss2.item())

        return (loss1 + loss2) / 2.0

class VideoData(Dataset):
    def __init__(self):
        self.n = 500
        self.user_ids = torch.randint(low=0, high=1000, size=(self.n, ))
        self.video_ids = torch.randint(low=0, high=1000, size=(self.n, ))
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.video_ids[idx]

    def __len__(self) -> int:
        return self.n


def warmup(step):
    if step < 1000:
        return step / 1000.0
    return 1.0

def train():
    model = TwoTowerModel()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
        betas=(0.95, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    dataset = VideoData()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    lossFn = ContrastiveLoss()
    epochs = 10

    for i in range(epochs):
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            user_ids, video_ids = data
            user_logits, video_logits = model((user_ids, video_ids))
            loss = lossFn(user_logits, video_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            # break
        
        avg_loss = total_loss / len(dataloader)
        print(f"after epoch {i}, loss = {avg_loss}")


if __name__ == "__main__":
    train()
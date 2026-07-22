"""
fraud detection problem
given some data
need to

create train/val/test data
    - downsample negatives or upsample positives
create model architecture
create loss function (bce ?)
have weighted loss

0.1% - similar to ads ranking

120$ cost for false negative
8$ cost for false positive
helps us set the precision/recall threshold

precision = TP / TP + FP
recall = TP / TP + FN

recall is preferred over precision - exactly how do
we determine the boundary threshold using the cost above?

training row
txn_id, is_fraud, category, currency, country, user_id, user_fraud_rate_30d,
user_mean_transaction_price_30d, user_std_transaction_price_30d

 - some categorical, some numeric features


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math

class FraudModel(nn.Module):
    MAX_CATEGORIES = 1000
    MAX_CURRENCIES = 120
    MAX_COUNTRIES = 196
    MAX_USER_ID = 1000000
    def __init__(self):
        super().__init__()

        input_dims = 4 + 4 + 4 + 4 + 3
        self.model = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.category = nn.Embedding(self.MAX_CATEGORIES, 4)
        self.currency = nn.Embedding(self.MAX_CURRENCIES, 4)
        self.country = nn.Embedding(self.MAX_COUNTRIES, 4)
        self.user_ids = nn.Embedding(self.MAX_USER_ID, 4)

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                sigma = (2.0 / fan_in) ** 0.5
                nn.init.trunc_normal_(m.weight, mean=0.0, std=sigma, a=-2.0*sigma, b=2.0*sigma)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    
    def forward(self, x):
        batch, s = x.shape
        """
        x0 is category
        """
        print("x.shape: ", x.shape)
        print("x[:, 0].shape: ", x[:, 0].to(torch.long).shape)
        category_embedding = self.category(x[:, 0].to(torch.long)) #b, 4
        currency_embedding = self.currency(x[:, 1].to(torch.long))
        country_embedding = self.country(x[:, 2].to(torch.long))
        user_id_embedding = self.user_ids(x[:, 3].to(torch.long))
        print("user_id_embedding.shape: ", user_id_embedding.shape)

        remaining_embedding = x[:, 4:]
        input = torch.cat([category_embedding, currency_embedding, country_embedding, user_id_embedding, remaining_embedding], dim=1)

        logits = self.model(input).squeeze()
        return logits

class FraudLoss(nn.Module):
    def __init__(self, pos_label_weight: float = 3.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_label_weight))    

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # logits: b
        # labels: b
        return self.bce(logits, labels)


class FraudDataset(Dataset):
    def __init__(self):
        self.n = 100000
        #create features and labels
        categories = torch.randint(low=0, high=FraudModel.MAX_CATEGORIES, size=(self.n, 1))
        currencies = torch.randint(low=0, high=FraudModel.MAX_CURRENCIES, size=(self.n, 1))
        countries = torch.randint(low=0, high=FraudModel.MAX_COUNTRIES, size=(self.n, 1))
        user_ids = torch.randint(low=0, high=FraudModel.MAX_USER_ID, size=(self.n, 1))
        other_features = torch.rand((self.n, 3)) * 10
        self.x = torch.cat([categories, currencies, countries, user_ids, other_features], dim=1)
        print(self.x.shape)
        pos_fraction = 1e-3
        pos_count = math.floor(pos_fraction * self.n)
        neg_count = self.n - pos_count
        self.y = torch.cat([torch.zeros(neg_count), torch.ones(pos_count)])
        print(self.y.shape)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.n


def warmup(step):
    if step < 1000:
        return step / 1000.0
    return 1.0

def train():
    fraud = FraudModel()
    lossFn = FraudLoss(pos_label_weight=5.0)
    optimizer = torch.optim.AdamW(fraud.parameters(), lr=3e-3, betas=(0.95, 0.98), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    fraud_dataset = FraudDataset()
    loader = DataLoader(fraud_dataset, batch_size=32, shuffle=True)

    epochs = 1
    for i in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = fraud(x)
            loss = lossFn(logits, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"epoch: {i}, loss = {avg_epoch_loss}")

    eval(fraud)

def eval(model: nn.Module):
    pass

if __name__ == "__main__":
    train()


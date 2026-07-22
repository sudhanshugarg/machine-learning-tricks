import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Tiny vocabulary for reproducibility
# ------------------------------------------------------------------
vocab = list("abcdefghijklmnopqrstuvwxyz ") + ["<PAD>", "<EOS>"]
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}
PAD_IDX = char2idx["<PAD>"]
EOS_IDX = char2idx["<EOS>"]
VOCAB_SIZE = len(vocab)


def tokenize(text: str):
    return [char2idx.get(c, char2idx[" "]) for c in text.lower()]


def detokenize(tokens):
    return "".join([idx2char[t] for t in tokens])


# ------------------------------------------------------------------
# Minimal decoder-only Transformer
# ------------------------------------------------------------------
class BrokenTransformerLM(nn.Module):
    WARMUP_MAX_STEPS = 1000
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=8, dim_feedforward=256, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=torch.sigmoid,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch, seq_len] token indices
        """
        b, s = x.shape
        tok = self.token_emb(x)  # [B, S, D]

        # pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        # print(pos.shape)
        pos = torch.arange(s, device=x.device)
        pos = self.pos_emb(pos)

        # Students often uncomment one or the other but not both
        out = tok + pos
        # out = tok

        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
        out = self.transformer(out, mask=mask, is_causal=True)

        logits = self.lm_head(out)  # [B, S, V]
        return logits


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class CharDataset(Dataset):
    def __init__(self, text, block_size=32):
        self.data = tokenize(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def warmup_scheduler(step: int) -> float:
    # print(f"step = {step}")
    if step < BrokenTransformerLM.WARMUP_MAX_STEPS:
        return float(step / BrokenTransformerLM.WARMUP_MAX_STEPS)
    else:
        step = BrokenTransformerLM.WARMUP_MAX_STEPS
    return 1.0

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TinyShakespeare excerpt (normally you'd load the full file)
    raw_text = (
        "to be or not to be that is the question " * 100
        + "all the world is a stage and all the men and women merely players " * 100
        + "now is the winter of our discontent made glorious summer " * 100
    )

    dataset = CharDataset(raw_text, block_size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BrokenTransformerLM(VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=warmup_scheduler)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)  # [B, S, V]

            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        for name, param in model.named_parameters():
            if ("in_proj_weight" in name or "linear1.weight" in name or name == "token_emb.weight") and param.grad is not None:
                print(f"{name}: {param.norm():.2f}, {param.grad.norm():.6f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Generation (should be coherent but isn't)
    # ------------------------------------------------------------------
    model.eval()
    prompt = tokenize("women and men ")
    input_ids = torch.tensor([prompt], dtype=torch.long).to(device)

    for _ in range(50):
        with torch.no_grad():
            logits = model(input_ids)  # [1, S, V]
            next_token_logits = logits[:, -1, :]  # [1, V]
            probs = F.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated = detokenize(input_ids[0].cpu().tolist())
    print(f"\nGenerated: {generated}")


if __name__ == "__main__":
    train()
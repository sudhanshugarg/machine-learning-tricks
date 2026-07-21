# Debugging a Transformer That Won't Learn

## The Scenario

You are training a small **decoder-only (GPT-style) transformer** for next-token prediction on the TinyShakespeare dataset. After 20 epochs, your training loss is stuck around 3.8 (close to random-guess level for a ~65-token vocabulary) and generated text is pure gibberish.

Your task:
1. **Diagnose** what is preventing the model from learning a proper language model.
2. **Explain** the root cause of each bug using ML theory (attention dynamics, information flow in autoregressive models, Adam optimization theory, etc.).
3. **Fix** the code and verify the model learns a coherent next-token prediction objective.

You are allowed to run the code, inspect attention maps, print gradient norms, and add debugging instrumentation. Treat this as a real debugging session.

---

## The Buggy Code

```python
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
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=256, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch, seq_len] token indices
        """
        b, s = x.shape
        tok = self.token_emb(x)  # [B, S, D]

        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        pos = self.pos_emb(pos)

        # Students often uncomment one or the other but not both
        # out = tok + pos
        out = tok

        mask = None  # No causal masking!
        out = self.transformer(out, mask=mask)

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

    dataset = CharDataset(raw_text, block_size=32)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BrokenTransformerLM(VOCAB_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)  # [B, S, V]

            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Generation (should be coherent but isn't)
    # ------------------------------------------------------------------
    model.eval()
    prompt = tokenize("to be ")
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
```

---

## Your Task

1. **Run the code** (or mentally trace it). Is loss decreasing? Does generated text make sense? What is happening?
2. **Systematically debug**. For each suspected issue:
   - What is the theoretical reason it breaks learning in a transformer?
   - What empirical evidence (attention maps, gradient norms, generated text) would confirm it?
   - What is the fix?
3. **Produce a corrected version** that produces coherent completions from the prompt `"to be "` and reaches training loss < 1.5 within 50 epochs on this toy dataset.

---

## Open-Ended Discussion Questions

After fixing the code, consider these:

1. **Causal Masking Theory**: In an autoregressive model, each position can only attend to previous positions. If you remove the causal mask during training but keep it during inference, why does the model fail at generation even though training loss was low? Use the concept of **distribution shift** and **exposure bias** to explain.

2. **Positional Encoding**: Why can a transformer (without recurrence or convolution) not distinguish `"ab"` from `"ba"` if positional encodings are removed? Explain using the permutation-invariance of self-attention.

3. **Adam Warmup**: Transformers are almost universally trained with a learning-rate warmup. Explain the theoretical reason: what happens to the second-moment estimates ($v_t$ in Adam) in the first few steps if you start with a large LR? How does this cause early training instability?

4. **Label Shifting for LM**: In next-token prediction, the input at position $i$ predicts the token at position $i+1$. If you instead train the model to predict the token at position $i$ from itself (no shift), what is the theoretical maximum cross-entropy loss the model can achieve, and why does this represent a trivial solution?

5. **Attention Pattern Debugging**: If generation still produces repetitive loops (e.g., `"the the the the..."`), what attention-pattern pathology would you look for? How would you diagnose it by inspecting attention weights?

---

## Deliverables

- A list of **all bugs** you found, ranked by severity.
- A **brief theoretical explanation** for why each bug kills learning in a transformer.
- **Empirical evidence** used to confirm each bug (prints, attention visualizations, generated text).
- The **fully corrected code** with comments marking each fix.

"""
Knowledge Distillation: training a small student to learn from a large teacher.

Runnable, fully-deterministic demonstration (no external data needed) that shows
*why* the teacher's soft outputs ("dark knowledge") beat one-hot labels.

The realistic setting we simulate:
  - A large TEACHER was trained once on a large, well-curated (clean) label set.
  - You must now ship a small STUDENT (here ~40x smaller). The labels YOU have to
    train the student on are cheap and NOISY (40% are wrong) -- a very common
    real-world situation (crowd labels, weak supervision, delayed/uncertain
    labels). But you can still run the frozen teacher on your inputs to get its
    soft output distribution.

We train two IDENTICAL small students:
  (a) baseline  -> the noisy HARD labels only (plain cross-entropy)
  (b) distilled -> KD loss: teacher's soft targets + the noisy hard labels

The distilled student wins because the teacher's soft probabilities encode the
true class structure, effectively "denoising" the corrupted hard labels. This is
the same mechanism that makes distillation work in general: the teacher's full
distribution carries more (and here, cleaner) information than a one-hot vector.

The KD loss is the heart of the problem -- see `DistillationLoss` and note:
  - F.kl_div expects LOG-probabilities as `input` and PROBABILITIES as `target`.
  - reduction='batchmean' is the mathematically correct KL reduction.
  - the soft term is multiplied by T^2 to keep gradients on scale.

Run:  python -P code.py       (the -P avoids a stdlib name clash with 'code.py')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Synthetic dataset: multi-modal classes -> non-linear decision boundaries
# ---------------------------------------------------------------------------
def make_dataset(n_per_class=1200, n_classes=10, n_features=12, n_modes=4,
                 noise=2.0, center_scale=3.0):
    """Each class is a union of several Gaussian sub-clusters ("modes") scattered
    around the space, so the classes interleave and no linear boundary separates
    them. A high-capacity teacher learns this structure well; the value of its
    soft targets is what we want to transfer to the small student."""
    modes = torch.randn(n_classes, n_modes, n_features) * center_scale
    xs, ys = [], []
    per_mode = n_per_class // n_modes
    for c in range(n_classes):
        for m in range(n_modes):
            x = modes[c, m] + noise * torch.randn(per_mode, n_features)
            xs.append(x)
            ys.append(torch.full((per_mode,), c, dtype=torch.long))
    X = torch.cat(xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


def add_label_noise(y, flip_prob, n_classes=10, seed=5):
    """Randomly reassign a fraction `flip_prob` of labels to a random class.
    Simulates the cheap/noisy labels available for student training."""
    g = torch.Generator().manual_seed(seed)
    y_noisy = y.clone()
    mask = torch.rand(len(y), generator=g) < flip_prob
    y_noisy[mask] = torch.randint(0, n_classes, (int(mask.sum()),), generator=g)
    return y_noisy


# ---------------------------------------------------------------------------
# 2. Models  (output RAW LOGITS -- softmax/temperature live in the loss)
# ---------------------------------------------------------------------------
class TeacherMLP(nn.Module):
    """Large, high-capacity teacher."""

    def __init__(self, in_dim=12, n_classes=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


class StudentMLP(nn.Module):
    """Small, low-capacity student -- SAME number of output logits as teacher."""

    def __init__(self, in_dim=12, n_classes=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


# ---------------------------------------------------------------------------
# 3. The distillation loss  (the heart of the problem)
# ---------------------------------------------------------------------------
class DistillationLoss(nn.Module):
    r"""
    L = alpha * T^2 * KL(p_teacher || p_student)   [both softened at temp T]
      + (1 - alpha) * CE(student_logits, y)        [hard labels at T = 1]
    """

    def __init__(self, temperature=4.0, alpha=0.9):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        T = self.T

        # --- soft term: KL between softened distributions --------------------
        # F.kl_div(input, target): input must be LOG-probs, target must be probs.
        #   input  = log_softmax(student_logits / T)   <- student, log-probs
        #   target = softmax(teacher_logits / T)        <- teacher, probs (detached)
        # reduction='batchmean' divides by the batch size (N), giving the true
        # mean KL per example. The default 'mean' would divide by N*K -- wrong.
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),   # log-probabilities
            F.softmax(teacher_logits / T, dim=1),       # probabilities
            reduction="batchmean",
        ) * (T * T)                                     # cancel 1/T^2 gradient shrink

        # --- hard term: standard cross-entropy on the available labels (T=1) -
        hard_loss = self.ce(student_logits, targets)

        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


# ---------------------------------------------------------------------------
# 4. Train / eval helpers
# ---------------------------------------------------------------------------
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_supervised(model, loader, epochs=50, lr=1e-3):
    """Plain cross-entropy training (used for the teacher and the baseline)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
    return model


def train_distilled(student, teacher, loader, epochs=50, lr=1e-3, T=4.0, alpha=0.9):
    """Train the student with the combined distillation loss."""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    kd = DistillationLoss(temperature=T, alpha=alpha)
    teacher.eval()  # frozen
    for _ in range(epochs):
        student.train()
        for x, y in loader:
            opt.zero_grad()
            with torch.no_grad():                     # no gradient into teacher
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = kd(student_logits, teacher_logits, y)
            loss.backward()
            opt.step()
    return student


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------------
def main():
    # Data --------------------------------------------------------------------
    X, y = make_dataset()
    n_train = int(0.8 * len(X))
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256)

    # Clean labels for the teacher; noisy labels for the student.
    y_tr_noisy = add_label_noise(y_tr, flip_prob=0.4)
    clean_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    noisy_loader = DataLoader(TensorDataset(X_tr, y_tr_noisy), batch_size=64, shuffle=True)

    # --- Teacher: trained on the clean, curated labels -----------------------
    torch.manual_seed(SEED)
    teacher = TeacherMLP()
    train_supervised(teacher, clean_loader, epochs=50)
    teacher_acc = accuracy(teacher, test_loader)

    # --- Baseline student: the NOISY hard labels only ------------------------
    torch.manual_seed(1)  # same init as the distilled student
    student_baseline = StudentMLP()
    train_supervised(student_baseline, noisy_loader, epochs=50)
    baseline_acc = accuracy(student_baseline, test_loader)

    # --- Distilled student: teacher soft targets + noisy hard labels ---------
    torch.manual_seed(1)  # identical initialization for a fair comparison
    student_distilled = StudentMLP()
    train_distilled(student_distilled, teacher, noisy_loader,
                    epochs=50, T=4.0, alpha=0.9)
    distilled_acc = accuracy(student_distilled, test_loader)

    # --- Report --------------------------------------------------------------
    tp, sp = count_params(teacher), count_params(student_baseline)
    print("=" * 62)
    print("KNOWLEDGE DISTILLATION RESULTS  (student labels are 40% noisy)")
    print("=" * 62)
    print(f"Teacher params : {tp:,}")
    print(f"Student params : {sp:,}   ({tp / sp:.1f}x smaller)")
    print("-" * 62)
    print(f"Teacher   (trained on CLEAN labels)     : {teacher_acc:.4f}")
    print(f"Student   (hard NOISY labels only)      : {baseline_acc:.4f}")
    print(f"Student   (distilled, T=4, alpha=0.9)   : {distilled_acc:.4f}")
    print("-" * 62)
    gap = teacher_acc - baseline_acc
    recovered = distilled_acc - baseline_acc
    print(f"Teacher - baseline gap                  : {gap:+.4f}")
    print(f"Gap recovered by distillation           : {recovered:+.4f}", end="")
    if gap > 1e-6:
        print(f"   ({100 * recovered / gap:.0f}% of the gap)")
    else:
        print()
    print("=" * 62)
    if distilled_acc > baseline_acc:
        print("PASS: the distilled student beats the hard-label baseline.")
        print("The teacher's soft targets 'denoise' the corrupted hard labels --")
        print("dark knowledge carries more correct signal than one-hot labels.")
    else:
        print("NOTE: no improvement this run -- try more epochs or tune T/alpha.")


if __name__ == "__main__":
    main()

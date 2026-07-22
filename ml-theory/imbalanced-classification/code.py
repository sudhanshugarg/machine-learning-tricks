"""
Designing a model for a severely imbalanced dataset (fraud detection).

Reference implementation for the design question in `problem.md`:
  * a model architecture for tabular fraud data, and
  * several imbalance-aware loss functions (BCE / class-weighted BCE / focal),
demonstrated on a synthetic imbalanced dataset so you can *see* what each
choice does to the metrics.

Run it directly:
    python code.py

Only requires torch + numpy. Metrics (precision/recall/F1/PR-AUC/ROC-AUC)
are implemented by hand, so there is no sklearn / scipy / matplotlib
dependency.

The key teaching points the run drives home:
  * the *loss doesn't move the ranking much* — all three losses reach a
    similar ROC-AUC / PR-AUC on a fixed dataset, because they converge to a
    similar decision boundary;
  * but the loss *does* move the *operating point at a fixed 0.5 threshold*:
    plain BCE lands conservative (high precision, low recall), heavy class
    weights land over-aggressive (low precision, high recall), and focal
    lands in between — usually with the best F1;
  * given *any* of the three, cost-optimal threshold tuning on a held-out
    set recovers the right operating point for your cost structure —
    because the *decision* is separable from the *score*. The 0.5
    threshold is only optimal when costs are symmetric and probabilities
    are calibrated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ----------------------------------------------------------------------
# 1. Synthetic imbalanced dataset
# ----------------------------------------------------------------------
def make_imbalanced_data(
    n_total: int = 100_000,
    prevalence: float = 0.01,    # 1% positive (raised from the real ~0.1%
                                 # so the demo metrics are stable in seconds;
                                 # the *techniques* are identical at 0.1%).
    n_features: int = 8,
    seed: int = 0,
):
    """Return (X, y) tensors for a binary classification problem.

    Majority class 0:  x ~ N(0, I)
    Minority class 1:  x ~ N(mu, I) with a clear-but-overlapping signal in a
    few coordinates, so the classes are learnable but NOT trivially
    separable — which makes the threshold tradeoff real rather than vacuous.

    At 1% prevalence and n=100k there are ~1000 positives: enough for stable
    train/val/test metrics while keeping the whole run to a few seconds on
    CPU.
    """
    g = torch.Generator().manual_seed(seed)
    n_pos = int(round(n_total * prevalence))
    n_neg = n_total - n_pos

    X_neg = torch.randn(n_neg, n_features, generator=g)

    mu = torch.zeros(n_features)
    mu[0:3] = 1.5    # clear signal in 3 features
    mu[3:5] = 0.8    # mild signal in 2 more
    X_pos = torch.randn(n_pos, n_features, generator=g) + mu

    X = torch.cat([X_neg, X_pos], dim=0)
    y = torch.cat([torch.zeros(n_neg), torch.ones(n_pos)])
    perm = torch.randperm(n_total, generator=g)
    return X[perm], y[perm]


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float().unsqueeze(1)  # [N, 1] for BCE

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------
# 2. Model architecture
# ----------------------------------------------------------------------
class FraudMLP(nn.Module):
    """An MLP for tabular fraud classification.

    Design choices, each motivated by the problem:
      * BatchNorm on the input and every hidden layer — tabular features
        have wildly different scales (amount in dollars vs. a count), and
        BN keeps activations well-scaled so we can use a larger, stable
        learning rate.
      * Dropout — the *positive* class is tiny, so the model overfits the
        minority examples easily; dropout regularizes.
      * A couple of residual (skip) connections — keep gradients healthy
        and let signal from the rare positives reach the input layer.
      * A single output logit (sigmoid via BCEWithLogits) — binary, and we
        need a probability downstream for threshold selection.
    """

    def __init__(self, n_features, hidden=64, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_bn(x)
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        h = h + self.drop(F.relu(self.bn2(self.fc2(h))))   # residual
        h = h + self.drop(F.relu(self.bn3(self.fc3(h))))   # residual
        return self.out(h)                                 # logits [N, 1]


# ----------------------------------------------------------------------
# 3. Loss functions
# ----------------------------------------------------------------------
def bce_loss(logits, targets):
    """Plain BCE — the naive baseline.

    Under severe imbalance the model minimizes this by predicting ~0 for
    everything (the marginal prior), so accuracy is ~99% and the model
    catches ~no fraud at a 0.5 threshold (see the run).
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def weighted_bce_loss(logits, targets, pos_weight):
    """Class-weighted BCE.

        L = -[ w_pos * y * log(p) + (1 - y) * log(1 - p) ]

    `pos_weight` (w_pos) reweights the *positive* (minority) term so the
    two classes contribute equally in expectation. The Bayes-consistent
    choice (so the weighted risk reflects the true prevalence) is
    w_pos = N_neg / N_pos. PyTorch's `pos_weight` multiplies only the
    positive term.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Focal loss (Lin et al., 2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where  p_t = p        if y = 1
              = 1 - p     if y = 0
    and p = sigmoid(logit). The (1 - p_t)^gamma factor down-weights *easy,
    well-classified* examples so the optimizer's gradient budget goes to
    the hard ones. gamma = 0 recovers (alpha-weighted) BCE.

    `alpha` is the class-balancing weight; with a rare positive class we
    set alpha > 0.5 to upweight the minority. Implemented with
    BCE-with-logits for a numerically stable log(sigmoid).
    """
    # Per-example -log(p_t), computed stably.
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t) ** gamma * ce
    return loss.mean()


# ----------------------------------------------------------------------
# 4. Metrics (implemented by hand — no sklearn needed)
# ----------------------------------------------------------------------
@torch.no_grad()
def classifier_metrics(logits, targets, threshold=0.5):
    """Confusion-matrix metrics at a given threshold.

    Squeezes both inputs first: the model emits logits of shape [N, 1]
    while targets are often [N]; without the squeeze the boolean `&`
    would broadcast to an [N, N] matrix and the counts would be garbage.
    """
    probs = torch.sigmoid(logits).squeeze()
    y = targets.squeeze()
    preds = (probs >= threshold).float()
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1,
               tp=tp, fp=fp, fn=fn, tn=tn)


@torch.no_grad()
def pr_auc(logits, targets):
    """Average-precision estimate of PR-AUC.

        AP = (1 / n_pos) * sum_{ranked positives k} Precision@k

    This is the metric that actually moves when you fix the loss for
    imbalance. Note the baseline for *random* ranking is AP ~= prevalence
    (0.01 here), NOT 0.5 — a subtle but important point at low prevalence.
    """
    scores = torch.sigmoid(logits).squeeze()
    y = targets.squeeze()
    order = torch.argsort(scores, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0).float()
    fp = torch.arange(1, len(y_sorted) + 1, device=y.device).float() - tp
    precision = tp / torch.clamp(tp + fp, min=1.0)
    n_pos = max(int(y.sum()), 1)
    ap = precision[y_sorted == 1].sum() / n_pos
    return ap.item()


@torch.no_grad()
def roc_auc(logits, targets):
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity, tie-corrected:

        AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

    Equivalent to P(score(positive) > score(negative)). More stable than
    PR-AUC at low prevalence but *less informative* under imbalance, because
    it is dominated by the enormous number of true negatives.
    """
    scores = torch.sigmoid(logits).squeeze()
    y = targets.squeeze().long()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    s_sorted = scores[order]
    y_sorted = y[order]
    # Average ranks within tie groups.
    ranks = torch.empty_like(s_sorted, dtype=torch.float)
    n = len(s_sorted)
    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0   # 1-indexed, averaged over [i, j)
        i = j
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.item()


# ----------------------------------------------------------------------
# 5. Training + evaluation harness
# ----------------------------------------------------------------------
def train_one(model, loader, loss_fn, epochs=5, lr=1e-3, clip=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for X, y in loader:
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            # Gradient clipping — important under imbalance: weighted/focal
            # losses produce large, spiky gradients from the rare positives.
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    m = classifier_metrics(logits, y, threshold=0.5)
    m["pr_auc"] = pr_auc(logits, y)
    m["roc_auc"] = roc_auc(logits, y)
    return m


def fmt(m):
    return (f"acc={m['acc']:.4f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} "
            f"PR-AUC={m['pr_auc']:.3f} ROC-AUC={m['roc_auc']:.3f} "
            f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")


def run_experiment(loss_name, loader, X_test, y_test, n_features,
                   pos_weight=None):
    torch.manual_seed(0)
    model = FraudMLP(n_features=n_features)
    if loss_name == "bce":
        lf = bce_loss
    elif loss_name == "weighted_bce":
        lf = lambda lo, ta: weighted_bce_loss(lo, ta, pos_weight)
    elif loss_name == "focal":
        lf = focal_loss
    else:
        raise ValueError(loss_name)
    model = train_one(model, loader, lf, epochs=5)
    m = evaluate(model, X_test, y_test)
    print(f"  [{loss_name:12s}] {fmt(m)}")
    return model


# ----------------------------------------------------------------------
# 6. Cost-optimal threshold selection
# ----------------------------------------------------------------------
@torch.no_grad()
def pick_threshold(logits, y, c_fn, c_fp):
    """Find the threshold minimizing total cost on validation data.

        TotalCost(tau) = c_fn * FN(tau) + c_fp * FP(tau)
        with  pred = 1  iff  score >= tau.

    Sweep distinct scores in *descending* order: start at "predict 0 for
    everything" (threshold above the max score), then lower the threshold
    so examples enter the predicted-positive set best-first. Each positive
    that enters drops the cost by c_fn; each negative that enters raises it
    by c_fp. The minimum is the operating point that best trades false
    alarms against missed fraud given the cost structure. Derivation in
    solution.md.
    """
    scores = torch.sigmoid(logits).squeeze()
    y = y.squeeze()
    order = torch.argsort(scores, descending=True)
    s_sorted = scores[order]
    y_sorted = y[order]
    n = len(y_sorted)

    # Start above the max score: predict 0 for everything.
    total_pos = int(y_sorted.sum())
    fn, fp = total_pos, 0
    best_cost = c_fn * fn + c_fp * fp
    best_t = 1.01   # "predict 0 for all" sentinel

    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            if y_sorted[j] == 1:
                fn -= 1      # was FN, now TP  -> cost drops by c_fn
            else:
                fp += 1      # was TN, now FP  -> cost rises by c_fp
            j += 1
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost = cost
            best_t = s_sorted[i].item()
        i = j
    return best_t, best_cost


def cost_of(m, c_fn, c_fp):
    return c_fn * m["fn"] + c_fp * m["fp"]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    X, y = make_imbalanced_data(n_total=100_000, prevalence=0.01, seed=0)
    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    pos_weight = n_neg_train / max(n_pos_train, 1)
    print(f"Train: {len(y_train)} ({n_pos_train} pos, {n_neg_train} neg) "
          f"-> prevalence {n_pos_train / len(y_train):.4%}")
    print(f"Val  : {len(y_val)} ({int((y_val==1).sum())} pos)")
    print(f"Test : {len(y_test)} ({int((y_test==1).sum())} pos)")
    print(f"Class-weight w_pos = N_neg/N_pos = {pos_weight:.1f}")
    print()

    train_ds = TabularDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # --- Compare loss functions (same architecture, 5 epochs each) ---
    print("Comparing loss functions (same architecture, 5 epochs, "
          "threshold 0.5):\n")
    run_experiment("bce", loader, X_test, y_test, X_train.shape[1])
    run_experiment("weighted_bce", loader, X_test, y_test,
                   X_train.shape[1], pos_weight=pos_weight)
    run_experiment("focal", loader, X_test, y_test, X_train.shape[1])

    # --- Resampling: WeightedRandomSampler for balanced batches ---
    print("\nWith WeightedRandomSampler (balanced batches) + focal loss:")
    w_pos = 1.0 / max(n_pos_train, 1)
    w_neg = 1.0 / max(n_neg_train, 1)
    sample_weights = torch.where(
        y_train == 1,
        torch.full_like(y_train, w_pos),
        torch.full_like(y_train, w_neg),
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train),
                                    replacement=True)
    balanced_loader = DataLoader(train_ds, batch_size=512, sampler=sampler)
    torch.manual_seed(0)
    model = train_one(FraudMLP(n_features=X_train.shape[1]),
                      balanced_loader, focal_loss, epochs=5)
    m = evaluate(model, X_test, y_test)
    print(f"  [focal+sample] {fmt(m)}")

    # --- Cost-optimal threshold on a CALIBRATED (plain-BCE) model ---
    # The theoretical tau* = c_FP/(c_FP+c_FN) only holds when the scores are
    # calibrated probabilities. Plain BCE is the MLE and is roughly
    # calibrated; weighted/focal losses distort the probabilities, which is
    # exactly why the theoretical 0.0625 doesn't match their empirical
    # optimum (see solution.md). So we demo the cost decision on the
    # plain-BCE model, whose empirical tau* lands near the theory.
    print("\nThreshold selection under asymmetric cost "
          "(c_FN=$120, c_FP=$8), on a calibrated plain-BCE model:\n")
    torch.manual_seed(0)
    model = train_one(FraudMLP(n_features=X_train.shape[1]),
                      loader, bce_loss, epochs=5)
    model.eval()  # use running BN stats / turn off dropout — same regime as
                 # `run_experiment`, so the test@0.5 numbers match it exactly
    with torch.no_grad():
        logits_val = model(X_val)
        logits_test = model(X_test)
    t_star, cost = pick_threshold(logits_val, y_val, c_fn=120.0, c_fp=8.0)
    n_val_samples = max(len(y_val), 1)
    print(f"  cost-optimal threshold on val: tau* = {t_star:.3f} "
          f"(expected cost/val-sample = ${cost / n_val_samples:.3f})")
    print(f"  theoretical  tau* = c_FP/(c_FP+c_FN) = "
          f"{8.0/(8.0+120.0):.3f}  (matches only if probs are calibrated)")
    m05 = classifier_metrics(logits_test, y_test, 0.5)
    mstar = classifier_metrics(logits_test, y_test, t_star)
    print(f"  test @ 0.5   : prec={m05['precision']:.3f} rec={m05['recall']:.3f} "
          f"f1={m05['f1']:.3f} (TP={m05['tp']} FP={m05['fp']} FN={m05['fn']}) "
          f"cost=${cost_of(m05, 120, 8):.0f}")
    print(f"  test @ tau*   : prec={mstar['precision']:.3f} rec={mstar['recall']:.3f} "
          f"f1={mstar['f1']:.3f} (TP={mstar['tp']} FP={mstar['fp']} FN={mstar['fn']}) "
          f"cost=${cost_of(mstar, 120, 8):.0f}")
    print("  -> tau* is far below 0.5 because a missed fraud ($120) costs"
          " ~15x a false\n     block ($8): we accept more false positives to"
          " drive down the far costlier false\n     negatives — and the"
          " total cost drops as a result.")


if __name__ == "__main__":
    main()

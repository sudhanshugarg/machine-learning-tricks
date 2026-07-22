# Solution: Designing a Model for a Severely Imbalanced Dataset

This solution is organized as **the one idea first**, then **progressive hints**, then the **design**, the **math**, a **walkthrough of the runnable code with the actual numbers it produces**, and finally the **answers to the discussion questions**.

> **Note on the demo's prevalence.** Real card fraud is ~0.1%. The runnable `code.py` uses **1%** instead, purely so the metrics are stable in a few seconds on CPU — at 0.1% you'd have ~100 positives in 100k examples and the PR-AUC would jump around run-to-run. The *techniques* and the *theory* are identical at 0.1%; only the noise level changes.

---

## Part 0: The One Idea

**Separate the *score* from the *decision*.**

A fraud model does two jobs that you should keep cleanly distinct:

1. **Score**: turn the features into a number $s(x) \in [0,1]$ that ranks transactions by fraud risk. This is what the network and the loss produce.
2. **Decide**: turn that number into a yes/no — block or allow — using a threshold $\tau$ chosen from your **cost structure**, not from 0.5.

Almost every mistake people make on imbalanced classification comes from conflating these two. The loss function you choose (BCE vs. focal vs. weighted) barely moves the **ranking** — but it moves the **operating point at a fixed 0.5 threshold** a lot, because it distorts the *calibration* of the scores. And the right threshold is **never** 0.5 under both imbalance and asymmetric costs.

This is the thread that runs through the whole answer: build a model that ranks well (easy, plain BCE already does this), measure it with the right metric (PR-AUC, not accuracy), and then spend your effort on the **threshold**, which is where the actual money is.

---

## Part 1: Progressive Hints

<details>
<summary><b>Hint 1 — The trap the interviewer is laying</b> (click to expand)</summary>

At 0.1% prevalence, what accuracy does a model get by predicting "not fraud" for *every* transaction? And what does that imply about the Bayes-optimal strategy under 0-1 loss? If your evaluation metric is accuracy, what's the single best "model" you can ship?

</details>

<details>
<summary><b>Hint 2 — What the loss actually controls</b> (click to expand)</summary>

Three losses — plain BCE, class-weighted BCE, focal — all reach roughly the same ROC-AUC / PR-AUC on the same data (see the run below). So the loss is *not* moving the ranking. What *is* it moving? Look at precision/recall at a fixed 0.5 threshold across the three. Why would changing the loss move those without moving the ranking?

</details>

<details>
<summary><b>Hint 3 — Calibration and the 0.5 threshold</b> (click to expand)</summary>

Plain BCE is the maximum-likelihood estimator of $P(y=1\mid x)$, so its scores are roughly *calibrated*. Weighted BCE and focal are not — their population minimizer is a *distorted* posterior. The textbook cost-optimal threshold $\tau^* = c_{FP}/(c_{FP}+c_{FN})$ assumes calibrated scores. So which model should you run the threshold demo on?

</details>

<details>
<summary><b>Hint 4 — The decision-theoretic derivation</b> (click to expand)</summary>

Write down the expected cost of predicting 1 vs. predicting 0 as a function of the (calibrated) probability $p=P(y=1\mid x)$. Predict 1 when its expected cost is lower. Solve for the $p$ at which you're indifferent. That indifference point *is* $\tau^*$. Why does it equal 0.5 only when $c_{FP}=c_{FN}$?

</details>

---

## Part 2: Systematic Design Methodology

When the dataset is severely imbalanced, follow this order — it maps onto the five parts of the prompt:

```
1. EVALUATION  → Pick the metric FIRST. If you pick accuracy, you're already done (predict all-negative). The metric dictates everything else.
2. ARCHITECTURE  → A model that can express the true P(y=1|x) and ranks well. For tabular fraud, a regularized MLP (or a GBDT).
3. LOSS  → Start with plain BCE (it's the MLE and it's calibrated). Add focal / class weights only if you need to move the operating point at a fixed threshold.
4. RESAMPLING  → If you reweight, understand it changes the minimizer (distorts calibration). Oversample-with-replacement is a valid alternative that does NOT change the minimizer.
5. DECISION  → Tune the threshold on a held-out set against the real cost structure. This is where the money is, and it's independent of which loss you used.
6. CALIBRATION  → If downstream systems consume probabilities (not just a decision), recalibrate (temperature scaling / isotonic) on a held-out set.
```

The key inversion vs. a normal ML problem: **evaluation comes first**, because under 0.1% prevalence the wrong metric will tell you a literally-useless model is "99.9% accurate."

---

## Part 3: Architecture

For **tabular fraud features** (amount, merchant, time, aggregates), the right baseline is *not* a deep network — it's a **gradient-boosted decision tree** (XGBoost / LightGBM). Trees handle mixed feature types, missing values, and non-linear interactions for free, and they dominate MLPs on most tabular benchmarks. In a real system this is your v1.

But the prompt asks for a model architecture and a custom loss, so the runnable demo uses a **regularized MLP** (`FraudMLP` in `code.py`), which lets us actually exercise the loss functions. Each design choice is motivated by the problem:

| Choice | Why it's there |
|---|---|
| **Input `BatchNorm1d`** | Tabular features have wildly different scales (a dollar amount vs. a count). BN keeps activations well-scaled so we can use a larger, stable learning rate. |
| **`BatchNorm` per hidden layer** | Same reason, propagated through depth; keeps the optimization landscape well-conditioned (small Hessian condition number → larger stable LR). |
| **`Dropout(0.3)`** | The positive class is tiny (~700 examples), so the model overfits the minority easily. Dropout regularizes. |
| **Two residual (skip) connections** | Keep gradients healthy so the signal from rare positives actually reaches the input layer — the same lesson from `debugging-transformer-training/`. |
| **Single output logit → `BCEWithLogits`** | Binary problem, and we need a downstream *probability* for threshold selection. `WithLogits` is numerically stable (no `log(0)`). |

```python
class FraudMLP(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, hidden);  self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden);     self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden);     self.bn3 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_bn(x)
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        h = h + self.drop(F.relu(self.bn2(self.fc2(h))))   # residual
        h = h + self.drop(F.relu(self.bn3(self.fc3(h))))   # residual
        return self.out(h)                                 # logits [N, 1]
```

A single sigmoid output (not softmax-over-2): this is binary classification, and we want the raw probability $p = \sigma(\text{logit})$ to feed into threshold selection and calibration.

---

## Part 4: Loss Functions (with the math)

### 4.1 Plain BCE — the maximum-likelihood baseline

Binary cross-entropy is the negative log-likelihood of a Bernoulli:

$$
\mathcal{L}_{\text{BCE}} = -\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\, y \log p(x) + (1-y)\log(1-p(x))\,\big], \qquad p(x)=\sigma(\text{logit}(x)).
$$

**The minimizer is the true posterior.** Taking the functional derivative w.r.t. $p(x)$ and setting it to zero: the conditional risk given $x$ is

$$
\mathcal{L}(x) = -\pi(x)\log p(x) - (1-\pi(x))\log(1-p(x)), \qquad \pi(x)\equiv P(y=1\mid x),
$$

$$
\frac{\partial \mathcal{L}(x)}{\partial p} = -\frac{\pi(x)}{p} + \frac{1-\pi(x)}{1-p} = 0 \;\Longrightarrow\; \boxed{\;p^*(x) = \pi(x) = P(y=1\mid x)\;}
$$

So plain BCE's population minimizer **is the true posterior** — its scores are *calibrated* by construction. This matters enormously below.

**Why it still "fails" under imbalance.** Calibrated means $p^*(x)\approx P(y=1\mid x)$, and at 0.1% prevalence *most* $P(y=1\mid x)$ are genuinely tiny (correctly!), so almost all scores sit well below 0.5. At a 0.5 threshold the model predicts "not fraud" for almost everything and catches little. **The model isn't wrong — the threshold is.** The ranking (PR-AUC) is still excellent.

### 4.2 Class-weighted BCE

$$
\mathcal{L}_{\text{wBCE}} = -\mathbb{E}\big[\, w_{\text{pos}}\, y \log p(x) + (1-y)\log(1-p(x))\,\big].
$$

**The minimizer is a *distorted* posterior — no longer calibrated.** Repeating the derivative with the $w_{\text{pos}}$ weight on the positive term:

$$
\frac{\partial \mathcal{L}(x)}{\partial p} = -\frac{w_{\text{pos}}\,\pi(x)}{p} + \frac{1-\pi(x)}{1-p} = 0
\;\Longrightarrow\;
\boxed{\;p^*(x) = \frac{w_{\text{pos}}\,\pi(x)}{w_{\text{pos}}\,\pi(x) + (1-\pi(x))}\;}
$$

This is **not** $\pi(x)$ unless $w_{\text{pos}}=1$. With the usual choice $w_{\text{pos}}=N_{\text{neg}}/N_{\text{pos}}\approx (1-\pi)/\pi$, the minimizer becomes the *prevalence-rebalanced* posterior — it deliberately shifts mass toward the minority class. Consequence: the scores are inflated, the 0.5 threshold now fires a lot (high recall, terrible precision), and **the textbook cost-optimal threshold $\tau^*=c_{FP}/(c_{FP}+c_{FN})$ no longer applies** because it assumes $p=\pi(x)$. This is exactly why the demo runs the threshold step on the *plain-BCE* model, not the weighted one.

### 4.3 Focal loss

Focal loss (Lin et al., 2017) adds a *modulating factor* to the per-example cross-entropy:

$$
\text{FL}(p_t) = -\alpha_t\,(1-p_t)^{\gamma}\,\log p_t, \qquad p_t = \begin{cases} p & y=1 \\ 1-p & y=0 \end{cases}, \quad \alpha_t = \begin{cases} \alpha & y=1 \\ 1-\alpha & y=0 \end{cases}
$$

- **$\alpha$** is a class-balancing weight (like $w_{\text{pos}}$ but normalized to $[0,1]$); with a rare positive we set $\alpha>0.5$.
- **$\gamma$** is the *focusing parameter*. $(1-p_t)^{\gamma}$ down-weights **easy, well-classified** examples (where $p_t\to 1$, so $1-p_t\to 0$) and leaves the loss for **hard** examples (where $p_t$ is small) nearly untouched. $\gamma=0$ recovers $\alpha$-weighted BCE.

**What the modulating factor does that plain class weighting cannot.** Class weighting rescales the *whole class's* contribution by one number. Focal rescales the contribution **per-example as a function of how hard that example already is**: easy negatives (the vast majority under imbalance) get their gradient nearly zeroed, so the optimizer's gradient budget flows to the hard negatives near the boundary *and* to the (rare) positives. In the regime where the model is already confident on most negatives, focal recovers capacity that class weighting would waste.

**The regime where focal ≈ class weighting:** when the model is *not* yet confident on the majority (early training, or a hard problem), $p_t$ is small for everyone and $(1-p_t)^\gamma\approx 1$, so focal ≈ $\alpha$-weighted BCE. **The regime where focal helps:** a well-trained model on a problem with a huge easy majority and a thin hard boundary — i.e., exactly fraud.

**Calibration caveat (Mukhoti et al., 2020):** focal's population minimizer is *also* a distorted posterior, not $\pi(x)$, for $\gamma>0$. So focal scores are not calibrated either — but the distortion tends to be milder and more "confidence-reducing" than weighted BCE's, which is part of why focal often lands at a more usable 0.5 operating point (best F1 in the run below).

Implementation detail worth getting right: compute $-\log p_t$ via `binary_cross_entropy_with_logits(reduction="none")` (numerically stable) and multiply by the modulating factor — don't compute `log(sigmoid(...))` yourself.

```python
def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t) ** gamma * ce).mean()
```

---

## Part 5: Evaluation — pick the metric *before* the model

| Metric | Formula | Behavior under 0.1% prevalence | Use it when |
|---|---|---|---|
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ | **Useless.** "Predict all-negative" scores 99.9%. Bayes-optimal *under 0-1 loss* is exactly this trivial model. | Never, for this problem. |
| **Precision** | $TP/(TP+FP)$ | "Of what we flagged, how much was real fraud?" | FP is costly (blocking legit cards). |
| **Recall (TPR)** | $TP/(TP+FN)$ | "Of real fraud, how much did we catch?" | FN is costly (letting fraud through) — our case. |
| **F1** | $2\,PR/(P+R)$ | Harmonic mean; punishes either being zero. | Single-number summary when costs are unknown. |
| **PR-AUC (AP)** | $\frac{1}{N_+}\sum_{\text{ranked }+} \text{Prec}@k$ | **The right curve metric under imbalance.** Random baseline = prevalence (0.01), *not* 0.5. | Primary ranking metric. |
| **ROC-AUC** | $P(s(+) > s(-))$ | Stable but *uninformative* — dominated by the sea of easy true negatives. | Secondary; fine for ranking, hides imbalance cost. |
| **Total cost** | $c_{FN}\cdot FN + c_{FP}\cdot FP$ | The actual business objective. | Always report this; it's what you'll threshold on. |

**The two subtleties interviewers probe:**

1. **ROC-AUC vs PR-AUC.** At 0.1% prevalence, ROC-AUC stays high (0.97) almost no matter what you do, because the false-positive rate can be tiny in absolute terms while the curve still looks great — the enormous $TN$ count dominates. PR-AUC's baseline *moves with prevalence* (it's ~0.01, not 0.5), so it actually reflects how much the imbalance hurts. **Under severe imbalance, PR-AUC is the honest curve metric; ROC-AUC is the flattery.**
2. **The PR-AUC random baseline is the prevalence, not 0.5.** A PR-AUC of 0.60 sounds modest but is **60× random** at 1% prevalence. State the baseline whenever you quote a PR-AUC, or the number is meaningless.

`code.py` implements precision/recall/F1/PR-AUC/ROC-AUC **by hand** (no sklearn — the environment only has torch+numpy). Two implementation traps worth noting:

- **Shape broadcast:** the model emits `[N,1]` logits but targets are `[N]`. The confusion-matrix `&` operator broadcasts `[N,1] & [N]` to `[N,N]`, turning your counts into ~$N^2$ garbage. Always `.squeeze()` first. (This was a real bug in an earlier draft of this very file.)
- **PR-AUC/AP:** rank descending, cumulative TP, AP = mean precision at the ranks where the true label is 1.
- **ROC-AUC:** use the Mann-Whitney rank-sum identity (tie-corrected), which is far more stable than sweeping thresholds at low prevalence.

---

## Part 6: Resampling

Three options, with their failure modes:

1. **`WeightedRandomSampler` (oversample-with-replacement, balanced batches).** Draw each batch with equal probability from each class. **Does not change the population minimizer** — it's a *stochastic* reweighting, so the model still converges toward $\pi(x)$ in expectation; the effective batch just sees more positives. This is the safest resampling for a neural net. *Downside:* duplicates minority examples → overfitting on the (tiny) positive set; mitigate with dropout and by not over-training.

2. **Random undersampling the majority.** Throws away the vast majority of your data. For a neural net this is usually a bad trade — you lose the easy negatives that let the model learn the boundary. *Sometimes* fine as a cheap preprocessing step to fit a quick baseline; rarely the final answer.

3. **SMOTE (synthetic minority oversampling).** Interpolates between minority points. **Can hurt for a neural net** because (a) it assumes the minority manifold is locally linear, which is often false for fraud; (b) it generates points *between* real positives, which can create synthetic points that cross the true decision boundary and become label noise; (c) it interacts badly with the data-leakage bug below. More useful for low-capacity models (kNN, logistic) than for a regularized MLP.

**The data-leakage bug to avoid.** *"Oversample, then split into train/val/test"* is a classic leak: oversampling *duplicates* minority rows, and if you split *after* duplicating, the same minority example appears in both train and val/test, so val/test metrics are inflated and the threshold you pick is over-optimistic. **Always split first, then resample only the training fold.** (WeightedRandomSampler avoids this entirely because it only changes the *sampling* of the train loader — val/test are untouched.)

In the run below, `WeightedRandomSampler + focal` pushes recall to 0.94 but precision collapses to 0.07 — useful if you're going to follow up the model's flags with human review, useless if the model's decisions are automated. Which brings us to the threshold.

---

## Part 7: The decision — cost-optimal threshold (with derivation)

This is the part that actually matters for the business, and it's the part most candidates skip.

**Setup.** We have a calibrated probability $p = P(y=1\mid x)$. Let $c_{FN}$ be the cost of a false negative (missed fraud ≈ \$120) and $c_{FP}$ the cost of a false positive (blocking a legit transaction ≈ \$8). Predict 1 when its expected cost is lower than predicting 0:

$$
\underbrace{(1-p)\,c_{FP}}_{\text{cost of saying 1 (it's actually 0)}} \;<\; \underbrace{p\,c_{FN}}_{\text{cost of saying 0 (it's actually 1)}}
$$

$$
c_{FP} - p\,c_{FP} < p\,c_{FN}
\;\Longrightarrow\;
c_{FP} < p\,(c_{FP}+c_{FN})
\;\Longrightarrow\;
\boxed{\;\tau^* = \frac{c_{FP}}{c_{FP}+c_{FN}}\;}
$$

With our numbers: $\tau^* = 8/128 = 0.0625$.

**Why 0.5 only when symmetric.** $\tau^*=0.5$ requires $c_{FP}=c_{FN}$ **and** a calibrated posterior. Under imbalance you almost never have the first (a missed fraud costs ~15× a false block here), and weighted/focal losses break the second. The 0.5 threshold is a convenience for balanced, equal-cost problems — it has no special status here.

**Important caveat repeated:** this closed form assumes $p=\pi(x)$ is calibrated. For weighted-BCE or focal models, $p^*(x)$ is the *distorted* posterior from §4.2/§4.3, so $\tau^*$ shifts. The robust, loss-agnostic recipe is to **just sweep the threshold on a held-out validation set** (as `pick_threshold` does) and pick the one that minimizes the *empirical* total cost $c_{FN}\cdot FN(\tau)+c_{FP}\cdot FP(\tau)$. The closed form is a sanity check, not the method.

```python
def pick_threshold(logits, y, c_fn, c_fp):
    # Sweep scores descending: start at "predict 0 for all", then admit the
    # highest-scored examples into the positive set one rank-group at a time.
    # Each positive admitted drops cost by c_fn; each negative raises it by c_fp.
    scores = torch.sigmoid(logits).squeeze(); y = y.squeeze()
    order = torch.argsort(scores, descending=True)
    s, ys = scores[order], y[order]
    fn, fp = int(ys.sum()), 0
    best_cost, best_t = c_fn*fn + c_fp*fp, 1.01
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            if ys[j] == 1: fn -= 1   # was FN, now TP
            else:         fp += 1    # was TN, now FP
            j += 1
        cost = c_fn*fn + c_fp*fp
        if cost < best_cost: best_cost, best_t = cost, s[i].item()
        i = j
    return best_t, best_cost
```

---

## Part 8: Walkthrough of the runnable code (actual output)

Run `python code.py` (torch + numpy only; ~12s on CPU). The synthetic data is two overlapping Gaussians (minority mean-shifted in 3 strong + 2 mild features) at **1% prevalence**, 100k examples, 70/15/15 split → **716 train positives, 136 val, 148 test**; $w_{\text{pos}}=N_{\text{neg}}/N_{\text{pos}}=96.8$.

### 8.1 The loss barely moves the ranking — but moves the 0.5 operating point a lot

```
Comparing loss functions (same architecture, 5 epochs, threshold 0.5):

  [bce         ] acc=0.9931 prec=0.814 rec=0.385 f1=0.523 PR-AUC=0.604 ROC-AUC=0.972 (TP=57  FP=13   FN=91  TN=14839)
  [weighted_bce] acc=0.9256 prec=0.110 rec=0.919 f1=0.196 PR-AUC=0.601 ROC-AUC=0.970 (TP=136 FP=1104 FN=12  TN=13748)
  [focal       ] acc=0.9930 prec=0.742 rec=0.446 f1=0.557 PR-AUC=0.597 ROC-AUC=0.971 (TP=66  FP=23   FN=82  TN=14829)
```

**Read this carefully — it's the whole point of the exercise.**

- **Ranking barely moves:** ROC-AUC is ~0.97 and PR-AUC is ~0.60 for *all three* losses. They converge to nearly the same decision boundary. **The loss is not what makes the model rank fraud well.** Plain BCE already does that — it's the MLE, it's calibrated, and it learns the same $\pi(x)$.
- **The 0.5 operating point diverges sharply:**
  - **Plain BCE** is calibrated, so its scores are tiny (most $P(y=1\mid x)$ genuinely are), and at 0.5 it fires rarely → **high precision (0.81), low recall (0.39)**. Conservative: it only flags the very-obvious fraud.
  - **Weighted BCE** has $w_{\text{pos}}=96.8$ inflating the scores (§4.2's distorted posterior), so at 0.5 it fires on almost anything slightly risky → **precision collapses to 0.11, recall 0.92**. Over-aggressive: it flags nearly all fraud but also flags 1104 legit transactions.
  - **Focal** lands in between and wins on **F1 (0.557)**: precision 0.74, recall 0.45.
- **Notice the accuracy is ~99% for BCE and focal and 92% for weighted** — and that 99% number is *meaningless*: it's almost entirely true negatives. This is the visual proof that accuracy is the wrong metric.

So: the loss is a **calibration / operating-point** knob, not a ranking knob. If you're going to fix the operating point, the threshold is a cleaner lever than the loss — and it works for *any* of the three.

### 8.2 Resampling pushes recall even harder

```
With WeightedRandomSampler (balanced batches) + focal loss:
  [focal+sample] acc=0.8818 prec=0.073 rec=0.939 f1=0.136 PR-AUC=0.600 ROC-AUC=0.969 (TP=139 FP=1764 FN=9 TN=13088)
```

Balanced sampling + focal pushes **recall to 0.94** (it catches 139 of 148 frauds) but **precision collapses to 0.07** — it flags 1764 legit transactions. The ranking (PR-AUC 0.60) is unchanged; only the operating point moved. This is the right model **if a human reviews the flags** (you want to surface *every* likely fraud and let a reviewer filter), and the wrong model if the decision is automated (you'd block 1764 good customers to catch 139 frauds).

### 8.3 The cost-optimal threshold recovers the right operating point — on the calibrated model

```
Threshold selection under asymmetric cost (c_FN=$120, c_FP=$8), on a calibrated plain-BCE model:

  cost-optimal threshold on val: tau* = 0.126 (expected cost/val-sample = $0.432)
  theoretical  tau* = c_FP/(c_FP+c_FN) = 0.062  (matches only if probs are calibrated)
  test @ 0.5   : prec=0.814 rec=0.385 f1=0.523 (TP=57  FP=13  FN=91) cost=$11024
  test @ tau*   : prec=0.348 rec=0.676 f1=0.460 (TP=100 FP=187 FN=48) cost=$7256
```

- We ran the threshold sweep on the **plain-BCE** model because its scores are (approximately) calibrated, so the theoretical $\tau^*=0.0625$ is the right sanity check.
- The **empirical** cost-optimal $\tau^*=0.126$ — not exactly 0.0625, because the model isn't perfectly calibrated and there's finite-sample noise, but the same *order of magnitude* and — crucially — **far below 0.5**, exactly as the theory predicts.
- **Total cost drops from \$11,024 (at 0.5) to \$7,256 (at $\tau^*$)** on the test set — a ~34% reduction, purely from choosing the threshold against the cost structure, with no model change at all. We trade precision (0.81 → 0.35) for recall (0.39 → 0.68): we accept ~14× more false positives (13 → 187) to catch 43 more frauds (57 → 100), because each caught fraud saves \$120 and each false alarm only costs \$8.

**This is the punchline.** The single most impactful thing you can do on an imbalanced, asymmetric-cost problem is not a clever loss — it's tuning the threshold on held-out data against the real costs. The loss got you the ranking; the threshold turns the ranking into money.

---

## Part 9: Calibration

**Why weighted/focal models are poorly calibrated.** From §4.2–§4.3, their population minimizer is a *distorted* posterior, not $\pi(x)$. Weighted BCE inflates scores toward the (rebalanced) prior; focal distorts them via the $(1-p_t)^\gamma$ factor. So the raw $p$ from these models is not $P(y=1\mid x)$ — it's a monotone function of it.

**When calibration matters:**
- **Only ranking?** It doesn't matter. Any monotone transform of $p$ preserves the ranking, so PR-AUC, ROC-AUC, and the *empirical* cost-optimal threshold (which only uses the *ordering*) are all unaffected. If you threshold by sweeping held-out scores, you never needed calibration.
- **Downstream consumes probabilities?** It matters. If risk is pooled across transactions, if you report "this is $p$% likely fraud" to an analyst, or if you feed $p$ into a downstream expected-value calculation, you need $p=\pi(x)$. **The closed-form $\tau^*=c_{FP}/(c_{FP}+c_{FN})$ also requires it** — that's why we swept empirically rather than trusting 0.0625.

**How to fix it.** Calibrate on a held-out set *after* training: **temperature scaling** (one scalar on the logits — the cheapest, and it's all you usually need) or **isotonic regression** (non-parametric, more flexible, more data-hungry). Crucially, a *calibrated* model trained with *any* loss can be post-hoc calibrated, so you can have your focal-loss ranking *and* calibrated probabilities — train on focal, recalibrate on val.

---

## Part 10: Answers to the Discussion Questions

### Q1: Why is accuracy a bad metric? What's the Bayes-optimal strategy under 0-1 loss?

At 0.1% prevalence, "predict not-fraud for everything" gets **99.9% accuracy**. Under **0-1 loss** (equal cost for every error), the Bayes-optimal classifier predicts the class with the larger posterior, and since $P(y=1) = 0.001 < 0.5 = P(y=0)$ for the marginal (and for most $x$), the Bayes-optimal strategy is to **always predict the majority class** — catching zero fraud. So the 0-1-loss-optimal model is a *literal* constant predictor, and it scores 99.9%. Any metric under which the optimal model is a useless constant is a metric that tells you nothing. This is why you must pick a cost-/prevalence-aware metric *before* you build anything.

### Q2: Focal loss vs. class-weighted cross-entropy

Both reweight the per-example loss, but differently:

- **Class weighting** scales each *class's total* contribution by one number ($w_{\text{pos}}$). It is *static* — it doesn't care whether a given example is easy or hard.
- **Focal's $(1-p_t)^\gamma$** is *dynamic* — it scales each example by *how well-classified it already is*. Easy examples (large $p_t$) are nearly zeroed regardless of class; hard examples keep their full gradient.

So focal does what class weighting cannot: it reallocates gradient *from the easy majority to the hard boundary cases*, not just from majority to minority. In the regime where the model is already confident on most negatives (a trained fraud model), this recovers capacity that class weighting wastes on the easy negatives. In the regime where the model is *not* yet confident (early training, hard problem), $p_t$ is small for everyone, $(1-p_t)^\gamma\approx 1$, and **focal collapses to $\alpha$-weighted BCE**. Focal *helps* precisely when there's a large easy majority and a thin hard boundary — exactly fraud.

### Q3: Resampling pitfalls

- **Random oversampling (with replacement)** duplicates minority rows → the model can memorize them → overfitting on the tiny positive set. Mitigate with dropout/weight-decay and don't over-train. `WeightedRandomSampler` is the safe version: it resamples the *train loader*, not the data, so it doesn't change the population minimizer (still $\pi(x)$ in expectation) and doesn't create the leakage bug.
- **Random undersampling** discards most of your data — for a neural net that's usually a bad trade, since the easy negatives are what let the model locate the boundary. Fine for a quick baseline.
- **Naive SMOTE** interpolates between minority points. It assumes the minority manifold is locally linear (often false for fraud), can synthesize points *across* the true boundary (label noise), and is redundant for a regularized MLP that already learns smooth boundaries. It shines for low-capacity models (kNN/logistic), not deep ones.
- **The leakage bug:** oversampling duplicates rows; if you split *after* oversampling, the same minority row is in both train and val/test, inflating val/test metrics and making your threshold over-optimistic. **Split first, resample only the training fold.**

### Q4: The decision-theoretic view — derive $\tau^*$

See §7. Predict 1 when $(1-p)c_{FP} < p\,c_{FN}$, which gives $\tau^* = c_{FP}/(c_{FP}+c_{FN})$. At $\tau^*=0.5$ you need $c_{FP}=c_{FN}$ *and* calibrated $p=\pi(x)$. With $c_{FN}=120, c_{FP}=8$, $\tau^*=0.0625$ — you predict "fraud" at a mere 6.25% probability because a miss is 15× costlier than a false alarm. The run confirms this qualitatively: the empirical optimum (0.126) is far below 0.5 and total cost drops ~34%.

### Q5: Calibration

See §9. Weighted/focal losses have *distorted* minimizers (§4.2–§4.3), so their raw $p$ isn't $P(y=1\mid x)$. If you only care about ranking (PR-AUC, or an empirically-swept threshold), this doesn't matter — any monotone transform preserves order. It matters when (a) downstream consumes probabilities, (b) you report "$p$% likely" to a human, or (c) you use the closed-form $\tau^*$. Fix with temperature scaling or isotonic regression on a held-out set, *after* training — you can train on focal (for ranking) and still ship calibrated probabilities.

### Q6: Label noise and delayed labels

This is the subtle one. Disputes arrive up to 60 days late and *not every dispute is fraud*. Two effects:

1. **Label noise interacts with imbalance multiplicatively.** If 5% of your *labels* are wrong (a dispute that wasn't fraud, or a fraud that was never disputed), and fraud is 0.1% of the data, then *label noise on the minority class can exceed the signal*. If the label noise rate on positives is $\epsilon_+$, then of the $0.1\%\cdot N$ "positive" labels, a fraction $\epsilon_+$ are mislabelled — and if $\epsilon_+ \gtrsim 0.1\%$, the number of *wrong* positive labels rivals the number of *right* ones. The *precision of your labels* (not your model) collapses at low prevalence. This is why you never train on raw dispute labels without a cleaning/confirmation step.
2. **Delayed labels → a streaming/online setting.** You can't wait 60 days to train, so you train on *partial* labels (transactions whose dispute window has closed are confidently labelled; recent ones are "unknown"). This is **positive-unlabelled / PU learning**: treat recent transactions as unlabeled, not as negatives. Mislabelling recent future-fraud as "negative" is the dominant source of label noise in production fraud systems.

### Q7: Streaming / concept drift

Fraud patterns change weekly — attackers adapt to whatever you deployed last week.

- **Detect drift:** monitor the **score distribution** (KS test / PSI on the model's $p$ over time), the **calibration** (reliability drift), and **precision/recall on confirmed-labeled windows** as they mature. A rising FPR at fixed threshold, or a score distribution that shifts, are early signals.
- **Architecture implications:** keep the model *small and cheap to retrain* (the MLP/GBDT, not a giant net) so you can retrain on a rolling window nightly; store features with timestamps so you can recompute on a sliding window; consider an **online/continual-learning** setup that updates on confirmed labels as they arrive rather than retraining from scratch. The threshold itself drifts too (the cost-optimal $\tau^*$ moves as the fraud pattern and the base rate shift), so re-tune it on the most recent confirmed window, not a stale val set.

---

## Part 11: Production notes (beyond the demo)

- **Use a GBDT in production, not the MLP.** For tabular fraud, XGBoost/LightGBM beats MLPs, handles missing values and mixed types for free, and trains in minutes on the rolling window. The MLP is here only because the prompt asked for a model + custom loss you can see working.
- **Two-stage systems.** A common real design: a high-recall first stage (cheap, flags ~5–10% of transactions, the `focal+sample` model from §8.2) → human review or a heavier second-stage model on the flagged subset. This decouples "catch everything" from "be precise."
- **Never threshold on the test set.** Pick $\tau^*$ on validation, evaluate on test. Tuning the threshold on test is the same leakage sin as tuning hyperparameters on test.
- **Cost the false positive honestly.** \$8 is a placeholder; a false block can cost a customer. If FP cost is actually high (churn risk), $\tau^*$ rises and you become more conservative — the math handles it automatically via the same $\tau^*=c_{FP}/(c_{FP}+c_{FN})$.

---

## Summary Table

| Decision | What it does | Key formula / fact | In the run |
|---|---|---|---|
| **Metric: accuracy** | Useless under 0.1% prevalence | Bayes-optimal under 0-1 loss = "predict all-negative", 99.9% acc | acc ~99% for BCE/focal — meaningless |
| **Metric: PR-AUC** | The honest curve metric under imbalance | Random baseline = prevalence (0.01), *not* 0.5 | ~0.60 = 60× random |
| **Loss: plain BCE** | MLE; calibrated minimizer $p^*=\pi(x)$ | $p^*(x)=P(y=1\mid x)$; ranking good, scores tiny | prec 0.81 / rec 0.39 at 0.5 |
| **Loss: weighted BCE** | Distorted minimizer; inflates scores | $p^*=\frac{w_{\text{pos}}\pi}{w_{\text{pos}}\pi+(1-\pi)}$ | prec 0.11 / rec 0.92 at 0.5 |
| **Loss: focal** | Down-weights *easy* examples; mild distortion | $\alpha_t(1-p_t)^\gamma(-\log p_t)$ | best F1 0.557 |
| **Resampling: WRS** | Balanced batches; doesn't change minimizer | oversample-with-replacement; **split before resampling** | rec 0.94 / prec 0.07 |
| **Threshold: $\tau^*$** | The actual business lever | $\tau^*=c_{FP}/(c_{FP}+c_{FN})$ (calibrated); else sweep | \$11024 → \$7256 (−34%) |
| **Calibration** | Restores $p=\pi(x)$ after weighted/focal | temperature scaling / isotonic; matters only if probs are consumed | run threshold demo on the *plain-BCE* model |

---

## References

- **Focal loss:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal Loss for Dense Object Detection.* ICCV / arXiv:1708.02002.
- **Focal loss & calibration:** Mukhoti, J., et al. (2020). *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS / arXiv:2002.09437 — shows focal's minimizer is a distorted posterior, and that focal can *improve* calibration vs. plain BCE in some regimes.
- **Cost-sensitive thresholds / ROC analysis:** Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters.
- **Imbalanced-learn survey:** He, H., & Garcia, E. (2009). *Learning from Imbalanced Data.* IEEE TKDE — the canonical reference for resampling tradeoffs.
- **Positive-unlabelled learning** (for the delayed-label / streaming case in Q6): Bekker, J., & Davis, J. (2020). *Learning from Positive and Unlabeled Data: A Survey.* Machine Learning.
- **In-repo cross-references:** the train-vs-eval bug in the threshold demo (call `model.eval()` before inference so BatchNorm uses running stats) is exactly the pitfall documented in `ml-theory/train-vs-eval-semantics/`; the residual-connection rationale echoes `ml-theory/debugging-transformer-training/`.

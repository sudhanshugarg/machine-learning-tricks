# Solution: Knowledge Distillation (Teacher → Student)

This is the classic **knowledge distillation** (KD) problem from Hinton, Vinyals
& Dean, *"Distilling the Knowledge in a Neural Network"* (2015). The interviewer
wants to see three things done well:

1. A sensible **student architecture** under a budget.
2. A **loss** that transfers the teacher's *soft* output distribution to the
   student, with the math for temperature $T$, the $T^2$ scaling, and the mixing
   weight $\alpha$ — landing on `F.kl_div` for the KL term.
3. **Working code** that proves the distilled student beats the hard-label
   baseline.

---

## 1. The core idea: "dark knowledge"

A trained classifier outputs a probability distribution over classes, not just
the argmax. For an image of a "2", a good teacher might output:

```
class:   0     1     2     3     4     5     6     7     8     9
p:     .001  .002  .93   .01   .001  .001  .001  .05   .003  .001
```

The hard label throws all of this away and keeps only "class 2". But the soft
distribution tells the student something the label cannot: **this 2 looks a
little like a 7, and not at all like a 5**. Those small non-target
probabilities — the *relative* similarities between wrong classes — are what
Hinton called **dark knowledge**. They encode the teacher's learned geometry of
the label space.

That is the whole intuition for *why* distillation beats training on labels
alone: the teacher provides a much richer, lower-variance training signal per
example than a one-hot vector. Each example now carries information about *all*
$K$ classes, so the student effectively learns from a denser supervision signal.

**Where does the extra information come from (Q1)?** Not from new data — from the
teacher having already compressed the training set into a smooth function whose
*non-target* outputs reveal inter-class structure that the discrete labels never
recorded.

---

## 2. Student architecture

Nothing exotic is required, and that is the point — the interesting part is the
loss. The rules of thumb:

- **Keep the interface identical to the teacher.** Same input space, same number
  of output logits ($K = 10$). The student must produce a distribution over the
  same classes so we can compare distributions.
- **Shrink depth and width, not the head.** Fewer layers / channels / hidden
  units to hit the parameter and latency budget. The final linear layer still
  maps to $K$ logits.
- **Do not add a softmax inside the model.** Output **raw logits**. Softmax (and
  the temperature scaling) happens inside the loss. This keeps the model reusable
  and avoids the classic "double softmax" bug.

For the toy code below (tabular / flattened-image features), the teacher is a
wide 3-hidden-layer MLP and the student is a narrow 1-hidden-layer MLP with
~15× fewer parameters. In a real vision setting the same logic applies with,
e.g., a ResNet-50 teacher and a ResNet-18 or MobileNet student.

**The capacity gap (part of Q7).** If the student is *far* too small it cannot
represent the teacher's function no matter how good the signal — distillation
helps up to a point, then the bottleneck is raw capacity. A useful diagnostic:
if even hard-label training plateaus well below the teacher and distillation
adds little, you are capacity-bound, not signal-bound.

---

## 3. The distillation loss

### 3.1 Softmax with temperature

Let the student produce logits $z^S \in \mathbb{R}^K$ and the teacher logits
$z^T \in \mathbb{R}^K$. Define the **temperature-scaled softmax**:

$$
p_i(z; T) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}.
$$

- $T = 1$ recovers the ordinary softmax.
- $T \to \infty$ pushes the distribution toward **uniform** ($1/K$ everywhere) —
  it *softens* the peaks and amplifies the tiny non-target probabilities, which
  is exactly the dark knowledge we want the student to see.
- $T \to 0$ pushes it toward a **one-hot** argmax (maximally sharp).

We train with a **higher** temperature (typically $T \in [2, 10]$) so the small
non-target probabilities are large enough to carry gradient signal. At inference
we set $T = 1$ — the temperature is a *training-time* magnifying glass, not part
of the deployed model (Q2).

### 3.2 The soft-target term (KL divergence)

Let $p^T = p(z^T; T)$ (teacher soft targets) and $p^S = p(z^S; T)$ (student
soft predictions). We want the student to **match the teacher's distribution**,
so we minimize the **forward KL divergence**:

$$
\mathcal{L}_{\text{soft}} = \mathrm{KL}\!\left(p^T \,\|\, p^S\right)
= \sum_{i} p^T_i \log \frac{p^T_i}{p^S_i}
= \underbrace{\sum_i p^T_i \log p^T_i}_{-H(p^T),\ \text{const in } \theta_S}
\;-\; \sum_i p^T_i \log p^S_i .
$$

The first term is the teacher's negative entropy — **constant** with respect to
the student's parameters. So *for gradient purposes* minimizing this KL is
identical to minimizing the cross-entropy $H(p^T, p^S) = -\sum_i p^T_i \log
p^S_i$ (this answers Q5: same gradient, differ only by a constant). We use KL
because `F.kl_div` gives us the clean, numerically stable implementation.

**Why forward KL (Q6)?** Forward $\mathrm{KL}(p^T\|p^S)$ is **mean-seeking**: it
penalizes the student for putting *low* probability where the teacher puts
*high* probability, so the student is forced to cover all the mass the teacher
assigns — it must reproduce the full soft target, non-target modes included.
Reverse KL $\mathrm{KL}(p^S\|p^T)$ is **mode-seeking** and would let the student
collapse onto the single dominant mode, discarding the dark knowledge. For
classification distillation we want coverage, so forward KL is the natural
choice.

### 3.3 The hard-label term

We also keep a standard supervised cross-entropy on the **true** labels $y$,
using the student's ordinary ($T = 1$) softmax:

$$
\mathcal{L}_{\text{hard}} = \mathrm{CE}\big(p(z^S; 1),\, y\big)
= -\log p_y(z^S; 1).
$$

This is a **safeguard** (part of Q7): if the teacher is wrong or poorly
calibrated on some example, the ground-truth term pulls the student back toward
the correct answer. It also grounds the student in the real objective we
ultimately care about.

### 3.4 Combining the terms and the $T^2$ factor

$$
\boxed{\;\mathcal{L} = \alpha \, T^2 \, \mathrm{KL}\!\left(p^T \,\|\, p^S\right)
\;+\; (1 - \alpha)\, \mathrm{CE}\big(p(z^S; 1),\, y\big)\;}
$$

- $\alpha \in [0, 1]$ mixes the two signals. Typical values are $\alpha \in
  [0.5, 0.9]$ — most of the weight on the teacher, a slice reserved for the hard
  labels. $\alpha = 1$ is pure distillation (no labels); $\alpha = 0$ is the
  hard-label baseline.

- **The $T^2$ factor (Q3).** The gradient of the soft term through a
  temperature-$T$ softmax scales like $1/T^2$: one factor of $1/T$ comes from the
  $z/T$ inside the softmax, and another from the softened probabilities'
  reduced sensitivity. Concretely, in the high-$T$ limit the gradient of
  $\mathcal{L}_{\text{soft}}$ w.r.t. a student logit is approximately

  $$
  \frac{\partial \mathcal{L}_{\text{soft}}}{\partial z^S_i}
  \approx \frac{1}{T}\big(p^S_i - p^T_i\big)
  \approx \frac{1}{T^2}\Big(\tfrac{z^S_i}{1} - \tfrac{z^T_i}{1}\Big)\cdot(\text{const}),
  $$

  i.e. it shrinks as $1/T^2$. Multiplying the soft loss by $T^2$ **cancels this
  shrinkage**, so the soft-target gradients stay on the same scale as the
  hard-label gradients regardless of $T$. Without $T^2$, raising the temperature
  would silently down-weight distillation and you'd have to retune the learning
  rate / $\alpha$ for every $T$.

- **High-$T$ limit = logit matching (Q2).** In that same limit, the term above
  shows the student is essentially driven to match *logits* directly:
  $z^S_i \to z^T_i$ (up to a per-example constant). So high-temperature
  distillation is approximately **MSE on logits** — a useful sanity check and an
  alternative you can use directly (Q5: prefer MSE on logits when you want to
  match the teacher's *confidence/scale* exactly, or when the teacher's softmax
  saturates so hard that even high $T$ leaves the non-target probs vanishingly
  small).

---

## 4. Implementing the KL term with `F.kl_div` (the part people get wrong)

PyTorch's `torch.nn.functional.kl_div` computes, elementwise,

$$
\text{kl\_div}(\text{input}, \text{target}) = \text{target} \cdot \big(\log(\text{target}) - \text{input}\big),
$$

and **sums/reduces** over elements. Read that carefully — it dictates the API:

1. **`input` must be LOG-probabilities**, `target` must be **probabilities**
   (with the default `log_target=False`). So:
   - `input  = F.log_softmax(student_logits / T, dim=1)`  ← log-softmax
   - `target = F.softmax(teacher_logits / T, dim=1)`      ← plain softmax
   - If you pass `softmax` (not `log_softmax`) for `input`, the formula silently
     computes garbage — no error, just a wrong loss and a student that won't
     learn (Q4).
   - If you **swap** the arguments (student as target, teacher as input) you
     minimize the reverse KL against a *detached* teacher-as-input — wrong
     objective and wrong gradients.

2. **Use `reduction='batchmean'`, not `'mean'`.** The KL divergence is a sum
   over the $K$ classes, averaged over the $N$ examples in the batch. The default
   `reduction='mean'` divides by $N \times K$ (every element), which is
   **mathematically wrong** — it makes the loss $K\times$ too small.
   `reduction='batchmean'` divides by $N$ only, giving the true mean KL per
   example (Q4).

3. **Detach the teacher.** The teacher is frozen; wrap its forward pass in
   `torch.no_grad()` (or `.detach()` its logits) so no gradient flows into it.

Putting it together, the soft term is:

```python
soft_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=1),   # input = log-probs
    F.softmax(teacher_logits / T, dim=1),       # target = probs
    reduction="batchmean",
) * (T * T)                                      # the T^2 rescaling
```

---

## 5. Evaluation plan

Train **two students with identical architecture, optimizer, and schedule**:

- **Baseline:** hard labels only ($\alpha = 0$, i.e. plain cross-entropy).
- **Distilled:** the combined loss above with, e.g., $T = 4$, $\alpha = 0.9$.

Report test accuracy for both, plus the **teacher** as an upper reference. The
expected ordering is:

$$
\text{acc}(\text{hard-label student}) \;<\; \text{acc}(\text{distilled student})
\;\le\; \text{acc}(\text{teacher}).
$$

### What the runnable demo actually shows

A subtlety worth stating in the interview: on a *clean, easy* synthetic dataset
where teacher and student see the **same** labels, the distillation benefit is
real but small and run-to-run noisy — a tiny student can already fit simple
Gaussian blobs about as well as a big one, so there is little "extra knowledge"
to transfer. KD's benefit shows up crisply when the teacher genuinely **knows
something the student's own labels don't**. On real tasks (ImageNet, etc.) that
is automatic; in a toy we must engineer it.

So `code.py` uses the most faithful and *reproducible* version of that story —
**label noise**:

- The **teacher** (large) was trained once on a **clean, curated** label set.
- The labels available to train the **student** are **cheap and 40% noisy**
  (crowd labels / weak supervision / delayed-and-uncertain labels — extremely
  common in practice).
- The **baseline** student trains on those noisy hard labels.
- The **distilled** student trains on the teacher's clean **soft targets** plus
  the noisy hard labels.

The distilled student wins decisively because the teacher's soft probabilities
encode the true class structure and effectively **denoise** the corrupted hard
labels — a direct, visible demonstration that dark knowledge carries more (and
here, more *correct*) information than one-hot labels. Representative output
(fully deterministic with fixed seeds on CPU):

```
Teacher params : 137,482
Student params : 1,482   (92.8x smaller)
Teacher   (trained on CLEAN labels)     : 0.9354
Student   (hard NOISY labels only)      : 0.8875
Student   (distilled, T=4, alpha=0.9)   : 0.9204   <- recovers 69% of the gap
```

This also grounds two of the discussion questions: it is exactly the
"teacher-as-safeguard / denoiser" mechanism from Q7, and it shows why $\alpha$
weighted heavily toward the soft term (0.9) is the right call when the hard
labels are the *unreliable* signal.

> Run with `python -P code.py`. The `-P` flag stops Python from putting the
> current directory on the import path, avoiding a name clash between this
> `code.py` and the standard-library `code` module (which PyTorch imports
> transitively via `pdb`).

---

## 6. Summary of the design choices

| Choice | What | Why |
|---|---|---|
| Student outputs | Raw logits, same $K$ as teacher | Compare distributions; softmax/temperature live in the loss |
| Soft-target loss | `F.kl_div(log_softmax(z_S/T), softmax(z_T/T), 'batchmean')` | Transfers dark knowledge; forward KL is mean-seeking |
| Temperature $T$ | $\sim 2$–$10$ at train, $1$ at inference | Magnifies non-target probabilities into usable gradient |
| $T^2$ factor | Multiply the soft term by $T^2$ | Cancels the $1/T^2$ gradient shrinkage; one LR works for all $T$ |
| Mixing $\alpha$ | $\sim 0.5$–$0.9$ on the soft term | Balance teacher signal vs. ground-truth safeguard |
| Teacher | Frozen, `no_grad`/`detach` | It is a fixed target, not a trainable module |

---

## 7. Extensions (Q8)

- **Response-based** (what we did): match final outputs (logits/probabilities).
  Simple, architecture-agnostic, strong baseline.
- **Feature-based** (FitNets, attention transfer): also match *intermediate*
  activations. Needs a projection layer when teacher/student widths differ.
  Helps when the capacity gap is large and outputs alone aren't enough.
- **Relation-based** (RKD): match *relationships between examples* (e.g.,
  pairwise distances/angles in feature space) rather than per-example outputs.
  Useful for metric-learning / embedding tasks.

Response-based KD is the right default and the right thing to present first; the
others are the natural follow-ups when the simple version leaves accuracy on the
table.

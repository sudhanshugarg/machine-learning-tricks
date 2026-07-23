# Knowledge Distillation — FAQ

Crisp answers to the follow-up questions an interviewer is likely to ask after
you present the design. See `problem.md` for the prompt and `solution.md` for the
full walkthrough.

---

### Q1. Why does distillation help at all? The teacher saw the same labels.

The extra information is not in the labels — it is in the **teacher's learned
function**. After training, the teacher outputs a smooth distribution over *all*
classes for every input. Its **non-target probabilities** encode inter-class
similarity ("this 2 resembles a 7, not a 5") that the one-hot label discarded.
Each training example now supplies information about all $K$ classes instead of
one, so the student learns from a **denser, lower-variance** signal. This is
"dark knowledge."

---

### Q2. What does the temperature $T$ do? Why high at train, $T=1$ at inference?

$p_i(z;T) = \dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$.

- $T \to 1$: ordinary softmax (sharp).
- $T \to \infty$: approaches **uniform** — it inflates the tiny non-target
  probabilities so they carry usable gradient.
- $T \to 0$: approaches a one-hot argmax.

We train at higher $T$ (typically 2–10) so the dark knowledge is visible to the
gradient. At inference we deploy $T = 1$ — temperature is a training-time
magnifying glass, not part of the shipped model. In the **high-$T$ limit** the
KL term reduces to matching logits directly ($z^S \approx z^T$), i.e. it becomes
approximately MSE on logits.

---

### Q3. Why multiply the soft loss by $T^2$?

The gradient of the soft term w.r.t. a student logit scales like $1/T^2$
(one $1/T$ from the $z/T$ inside the softmax, one from the flattened
probabilities). Concretely, in the high-$T$ limit
$\partial \mathcal{L}_{\text{soft}} / \partial z^S_i \approx \frac{1}{T^2}(z^S_i
- z^T_i)\cdot\text{const}$. Multiplying by $T^2$ **cancels that shrinkage**, so
the soft- and hard-target gradients stay on the same scale and a single learning
rate / single $\alpha$ works across temperatures. Without it, raising $T$
silently down-weights distillation.

---

### Q4. `F.kl_div` gotchas.

`F.kl_div(input, target)` computes `target * (log target − input)` and reduces.
So:

- **`input` must be log-probabilities**: `F.log_softmax(student_logits/T, dim=1)`.
- **`target` must be probabilities**: `F.softmax(teacher_logits/T, dim=1)`.
- Pass `softmax` (not `log_softmax`) for `input` → silently wrong loss, no error,
  student won't learn.
- **Swap the arguments** (student as `target`) → you optimize the wrong
  objective with wrong gradients.
- Use **`reduction='batchmean'`**, not the default `'mean'`. KL is a *sum over
  classes*, averaged over *examples*. `'mean'` divides by $N\times K$ (making the
  loss $K\times$ too small); `'batchmean'` divides by $N$, the correct KL.
- **Detach the teacher** (`torch.no_grad()`), it is frozen.

---

### Q5. KL vs. cross-entropy vs. MSE — same gradient?

$\mathrm{KL}(p^T\|p^S) = H(p^T,p^S) - H(p^T)$. The teacher entropy $H(p^T)$ is
**constant** w.r.t. the student, so minimizing KL and minimizing cross-entropy
$H(p^T,p^S)$ give the **identical gradient**. We use KL only because `F.kl_div`
is the clean, stable implementation.

Prefer **MSE on logits** when you want the student to match the teacher's
*confidence/scale* exactly, or when the teacher's softmax is so saturated that
even high $T$ leaves the non-target probabilities vanishingly small — then
matching logits directly transfers more than matching near-degenerate
distributions.

---

### Q6. Forward vs. reverse KL — which and why?

We minimize **forward** KL $\mathrm{KL}(p^T \| p^S)$ (teacher first).

- **Forward KL is mean-seeking:** it heavily penalizes the student for putting
  low probability where the teacher put high probability, forcing the student to
  **cover all** the teacher's mass — including the non-target modes that *are*
  the dark knowledge.
- **Reverse KL $\mathrm{KL}(p^S \| p^T)$ is mode-seeking:** it lets the student
  collapse onto the single dominant mode and ignore the rest — discarding
  exactly the signal we want to transfer.

For classification distillation we want coverage, so forward KL is the natural
choice.

---

### Q7. When does distillation break?

- **Bad/miscalibrated teacher:** the student inherits its mistakes. The
  hard-label term $(1-\alpha)$ is the **safeguard** — it keeps pulling the
  student toward ground truth where the teacher is wrong. (Symmetrically, when
  the *hard labels* are the unreliable signal — e.g. noisy crowd labels — you
  lean $\alpha$ toward the teacher; that is exactly the setup in `code.py`, where
  the teacher's clean soft targets **denoise** 40%-corrupted labels.)
- **Capacity gap:** if the student is *far* too small it physically cannot
  represent the teacher's function; soft targets can't fix a representational
  bottleneck and, with $\alpha$ high, may even hurt by down-weighting the labels.
  Diagnostic: if both hard-label and distilled training plateau far below the
  teacher, you are capacity-bound, not signal-bound — grow the student or use
  feature-based KD.

---

### Q8. Beyond logits — other kinds of distillation.

- **Response-based** (what we did): match final outputs
  (logits/probabilities). Simple, architecture-agnostic, strong default.
- **Feature-based** (FitNets, attention transfer): also match *intermediate
  activations*; needs a projection when widths differ. Helps when the capacity
  gap is large and outputs alone don't transfer enough.
- **Relation-based** (RKD): match *relationships between examples* (pairwise
  distances/angles in feature space) rather than per-example outputs. Good for
  metric-learning / embedding tasks.

Present response-based first; reach for the others when the simple version leaves
accuracy on the table.

---

### Q9. Practical defaults?

- **Temperature:** $T \in [2, 10]$; 3–4 is a good start.
- **Mixing:** $\alpha \in [0.5, 0.9]$ on the soft term; go higher when hard
  labels are noisy, lower when the teacher is weak.
- **Always** `reduction='batchmean'` and the $T^2$ factor.
- **Model outputs logits**, never an internal softmax — temperature and softmax
  belong in the loss.
- **Freeze the teacher** and run it under `torch.no_grad()`.
- At **inference**, drop the teacher and set $T=1$.

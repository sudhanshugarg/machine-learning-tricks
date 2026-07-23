# Training a Small Student Model to Learn From a Large Teacher

## The Scenario

You have trained (or been handed) a **large, accurate "teacher" model** for a
10-class image classification task. It works great, but it is far too big and
slow to deploy: it has hundreds of millions of parameters, needs a GPU to hit
your latency budget, and will never fit on the mobile devices where the model
actually has to run.

Your job is to ship a **small "student" model** — roughly 10–50× smaller — that
runs on-device within a tight latency and memory budget, while giving up as
little accuracy as possible.

The obvious baseline is to train the small model from scratch on the same
labeled data. When you try this, the small model underperforms: the gap to the
teacher is large. You suspect that the teacher "knows" more than the one-hot
labels can express, and you want the student to learn from the **teacher's full
output distribution**, not just the hard labels.

You are given:

- A trained **teacher** model $f_T$ (frozen — you will not update its weights).
- A labeled training set $\{(x_i, y_i)\}$ where $y_i \in \{0, \dots, 9\}$.
- A fixed **parameter / latency budget** for the student $f_S$.

---

## The Single Prompt

> Design a training procedure that lets a small **student** model learn from a
> large **teacher** model, so that the student is much cheaper to run but nearly
> as accurate. In particular, **design the student architecture and the loss
> function** (with the math): explain precisely what signal you extract from the
> teacher, how you combine it with the ground-truth labels, and why that signal
> is more informative than the hard labels alone. Then provide **working PyTorch
> code** for the student model and the distillation loss, and demonstrate that
> the distilled student beats an identical student trained only on hard labels.

You may make reasonable assumptions and state them. Treat this as a design
conversation — the interviewer will push on every choice, especially the loss.

---

## Your Task

1. **Architecture.** Propose a small student architecture appropriate for the
   budget. Justify the capacity trade-offs. Explain what stays the same between
   teacher and student (input space, number of classes / logits) and what
   changes (depth, width, parameter count).

2. **The distillation signal.** Define exactly what you take from the teacher.
   Why are the teacher's **soft probabilities** ("dark knowledge") more
   informative than one-hot labels? Introduce the **temperature** $T$ and derive
   what it does to the softmax distribution.

3. **Loss function.** Design the distillation loss **with the math**. It should
   combine (a) a term that matches the student's softened distribution to the
   teacher's softened distribution, and (b) a standard supervised term on the
   true labels. State which divergence you use for (a), why, and derive the
   gradient intuition. Explain the $T^2$ scaling factor and the mixing weight
   $\alpha$.

4. **Working code.** Provide runnable PyTorch code for the student model and a
   custom distillation loss. Your KL-divergence term should be implemented with
   `torch.nn.functional.kl_div` — be explicit about the `log_softmax` /
   `softmax` argument convention and the `reduction` you choose, since these are
   the two things people get wrong.

5. **Evaluation.** Show that the distilled student outperforms the same student
   trained on hard labels alone (same architecture, same budget, same schedule).
   Report the teacher accuracy as an upper reference.

---

## Open-Ended Discussion Questions

After you present your design, expect follow-ups like:

1. **Why does distillation help at all?** The teacher was trained on the same
   labels the student already has. Where does the extra information come from?
   What do the teacher's *wrong-class* probabilities encode?

2. **The temperature.** What happens to the soft targets as $T \to 1$? As
   $T \to \infty$? Why do we use a *higher* temperature during training but
   $T = 1$ at inference? Sketch how the KL term behaves in the high-$T$ limit and
   why that connects distillation to plain logit matching (MSE on logits).

3. **The $T^2$ factor.** Why is the KL term multiplied by $T^2$? Show what
   happens to the gradient magnitude of the soft-target term as $T$ grows, and
   explain how $T^2$ keeps the two loss terms on a comparable scale so a single
   learning rate works for both.

4. **`F.kl_div` gotchas.** `F.kl_div(input, target)` expects `input` to be
   **log-probabilities** and `target` to be **probabilities** (by default).
   What goes wrong if you pass probabilities for `input`, or if you swap the
   arguments? Why is `reduction='batchmean'` the mathematically correct choice
   rather than the default `reduction='mean'`?

5. **KL vs. cross-entropy vs. MSE.** Minimizing $\mathrm{KL}(p_T \| p_S)$ over
   the student differs from cross-entropy $H(p_T, p_S)$ only by the teacher's
   entropy $H(p_T)$, which is constant w.r.t. the student. So do they give the
   same gradient? When would you prefer MSE on logits instead of KL on
   softened probabilities?

6. **Which direction of KL?** We minimize the *forward* KL
   $\mathrm{KL}(p_T \| p_S)$ (teacher first). What would the *reverse* KL
   $\mathrm{KL}(p_S \| p_T)$ optimize instead (mode-seeking vs. mean-seeking),
   and why is forward KL the natural choice for classification distillation?

7. **When does it break?** What if the teacher is poorly calibrated or simply
   wrong on some examples — does the student inherit those mistakes? What is the
   role of the hard-label term $\alpha$ as a safeguard? What happens if the
   student's capacity is *far* too small (the "capacity gap")?

8. **Beyond logits.** This is *response-based* distillation (match the outputs).
   Briefly contrast it with *feature-based* distillation (match intermediate
   activations) and *relation-based* distillation. When would you reach for
   those instead?

---

## Deliverables

- A **student architecture** with justification for the capacity trade-offs.
- A **loss-function design** with the math for the softened-KL term, the
  hard-label term, and the $T$ / $T^2$ / $\alpha$ hyperparameters.
- **Working PyTorch code** (student model + custom distillation loss built on
  `F.kl_div`) that trains a student two ways — hard-label-only vs. distilled —
  and prints the accuracy gap on a held-out set, with the teacher as reference.
- An **evaluation** demonstrating the distilled student wins.

"""
Runnable Code Examples: Train vs Eval Semantics

Run this file directly to see concrete demonstrations of:
1. Dropout stochasticity in train vs determinism in eval.
2. BatchNorm running-stat updates.
3. LayerNorm invariance to mode.
4. Transformer attention dropout behavior.
5. Deterministic inference test pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def demo_dropout_behavior():
    """Show that Dropout is stochastic in train and identity in eval."""
    print("=" * 60)
    print("DEMO: Dropout Behavior")
    print("=" * 60)

    drop = nn.Dropout(p=0.5)
    x = torch.ones(1, 10)

    drop.train()
    out_train = torch.stack([drop(x) for _ in range(20)])
    variance_train = out_train.var(dim=0).mean().item()

    drop.eval()
    out_eval = torch.stack([drop(x) for _ in range(20)])
    variance_eval = out_eval.var(dim=0).mean().item()

    print(f"Train mode variance across 20 runs: {variance_train:.4f}")
    print(f"Eval mode variance across 20 runs:  {variance_eval:.6f}")
    print("Expected: train > 0, eval ≈ 0\n")


def demo_batchnorm_stats():
    """Show that BatchNorm updates running stats in train and freezes in eval."""
    print("=" * 60)
    print("DEMO: BatchNorm Running Stats")
    print("=" * 60)

    bn = nn.BatchNorm1d(4)
    # Initialize with non-trivial stats so we can see change
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

    prev_mean = bn.running_mean.clone()
    prev_var = bn.running_var.clone()
    prev_num = bn.num_batches_tracked.clone()

    # Train mode with batch size 8
    bn.train()
    x = torch.randn(8, 4) * 5 + 3  # mean ≈ 3, std ≈ 5
    out = bn(x)

    print("After TRAIN forward pass:")
    print(f"  running_mean changed? {not torch.equal(bn.running_mean, prev_mean)}")
    print(f"  running_var changed?  {not torch.equal(bn.running_var, prev_var)}")
    print(f"  num_batches_tracked:  {bn.num_batches_tracked.item()} (was {prev_num.item()})")

    # Reset trackers
    prev_mean = bn.running_mean.clone()
    prev_var = bn.running_var.clone()
    prev_num = bn.num_batches_tracked.clone()

    # Eval mode forward
    bn.eval()
    with torch.no_grad():
        out = bn(x)

    print("\nAfter EVAL forward pass:")
    print(f"  running_mean changed? {not torch.equal(bn.running_mean, prev_mean)}")
    print(f"  running_var changed?  {not torch.equal(bn.running_var, prev_var)}")
    print(f"  num_batches_tracked:  {bn.num_batches_tracked.item()} (was {prev_num.item()})")
    print()


def demo_layernorm_invariance():
    """Show that LayerNorm is unaffected by train/eval mode."""
    print("=" * 60)
    print("DEMO: LayerNorm Invariance")
    print("=" * 60)

    ln = nn.LayerNorm(10)
    x = torch.randn(2, 10)

    ln.train()
    out_train = ln(x)

    ln.eval()
    out_eval = ln(x)

    print(f"Train vs Eval outputs identical? {torch.allclose(out_train, out_eval)}")
    print("Expected: True\n")


def demo_transformer_attention_dropout():
    """Show that Transformer attention dropout is mode-dependent."""
    print("=" * 60)
    print("DEMO: Transformer Attention Dropout")
    print("=" * 60)

    embed_dim = 16
    num_heads = 2
    mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.5, batch_first=True)

    x = torch.randn(2, 5, embed_dim)  # batch=2, seq=5, dim=16

    mha.train()
    attn_outputs_train = []
    for _ in range(10):
        out, weights = mha(x, x, x, average_attn_weights=True)
        attn_outputs_train.append(out)
    variance_train = torch.stack(attn_outputs_train).var(dim=0).mean().item()

    mha.eval()
    with torch.no_grad():
        attn_outputs_eval = []
        for _ in range(10):
            out, weights = mha(x, x, x, average_attn_weights=True)
            attn_outputs_eval.append(out)
        variance_eval = torch.stack(attn_outputs_eval).var(dim=0).mean().item()

    print(f"Train mode attention output variance: {variance_train:.4f}")
    print(f"Eval mode attention output variance:  {variance_eval:.6f}")
    print("Expected: train > 0, eval ≈ 0\n")


def test_deterministic_inference(model: nn.Module, sample_input: torch.Tensor) -> bool:
    """
    Reusable test: verify that inference is deterministic.
    Returns True if deterministic, False otherwise.
    """
    model.eval()
    with torch.inference_mode():
        out1 = model(sample_input)
        out2 = model(sample_input)
    deterministic = torch.allclose(out1, out2, atol=1e-6)
    print(f"Deterministic inference test passed? {deterministic}")
    return deterministic


def demo_state_dict_contents():
    """Inspect what BatchNorm buffers are included in state_dict."""
    print("=" * 60)
    print("DEMO: State Dict Contents")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.Dropout(0.3),
        nn.Linear(20, 1),
    )

    print("Keys in state_dict:")
    for k in model.state_dict().keys():
        print(f"  - {k}")
    print("\nNotice: BatchNorm contributes running_mean, running_var, num_batches_tracked.")
    print("Dropout has no parameters or buffers.\n")


def main():
    print("RUNNING ALL DEMOS\n")
    demo_dropout_behavior()
    demo_batchnorm_stats()
    demo_layernorm_invariance()
    demo_transformer_attention_dropout()
    demo_state_dict_contents()

    # Final integration-style test
    print("=" * 60)
    print("INTEGRATION TEST: Deterministic Inference")
    print("=" * 60)
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 2),
    )
    sample = torch.randn(4, 10)
    test_deterministic_inference(model, sample)


if __name__ == "__main__":
    main()

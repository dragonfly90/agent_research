"""
E2E-TTT (End-to-End Test-Time Training) — PyTorch Version

Based on: "End-to-End Test-Time Training for Long Context"
Paper:   https://test-time-training.github.io/e2e.pdf
arXiv:   https://arxiv.org/abs/2512.23675
Official: https://github.com/test-time-training/e2e (JAX)

Core ideas:
1. Sliding-window causal attention (local context)
2. Dual MLP blocks (adaptive + frozen)
3. Test-time training via mini-batch SGD on next-token prediction
4. Meta-learning the initialization W_0

Usage:
    pip install torch
    python3 ttt_e2e_torch.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Multi-head attention restricted to a local sliding window."""

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build sliding-window causal mask
        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0.0

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale + mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class DualMLP(nn.Module):
    """
    Dual MLP from E2E-TTT.
    - adaptive_mlp: updated during test-time training
    - frozen_mlp:   stays frozen to preserve pre-trained knowledge
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        half_ff = d_ff // 2
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(d_model, half_ff, bias=False),
            nn.GELU(),
            nn.Linear(half_ff, d_model, bias=False),
        )
        self.frozen_mlp = nn.Sequential(
            nn.Linear(d_model, half_ff, bias=False),
            nn.GELU(),
            nn.Linear(half_ff, d_model, bias=False),
        )

    def forward(self, x):
        return self.adaptive_mlp(x) + self.frozen_mlp(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, window_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = DualMLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# E2E-TTT Model
# ---------------------------------------------------------------------------

class E2ETTT(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=4,
                 d_ff=128, max_seq_len=512, window_size=32,
                 ttt_batch_size=16, ttt_lr=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ttt_batch_size = ttt_batch_size
        self.ttt_lr = ttt_lr

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, window_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # TTT: last 25% of blocks
        n_ttt = max(1, n_layers // 4)
        self.ttt_block_indices = list(range(n_layers - n_ttt, n_layers))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def get_ttt_params(self):
        params = []
        for i in self.ttt_block_indices:
            params.extend(self.blocks[i].mlp.adaptive_mlp.parameters())
        return params

    def save_ttt_state(self):
        return {name: p.clone() for name, p in self.named_parameters()
                if any(f"blocks.{i}.mlp.adaptive_mlp" in name
                       for i in self.ttt_block_indices)}

    def restore_ttt_state(self, state):
        for name, p in self.named_parameters():
            if name in state:
                p.data.copy_(state[name])

    def ttt_adapt(self, context_ids):
        """
        Test-time training: mini-batch SGD on next-token prediction.
        Only updates adaptive MLP params in TTT blocks.
        """
        # Freeze everything except adaptive MLPs
        for name, p in self.named_parameters():
            is_ttt = any(f"blocks.{i}.mlp.adaptive_mlp" in name
                         for i in self.ttt_block_indices)
            p.requires_grad_(is_ttt)

        ttt_params = self.get_ttt_params()
        optimizer = torch.optim.SGD(ttt_params, lr=self.ttt_lr)

        T = context_ids.shape[1]
        n_batches = max(1, T // self.ttt_batch_size)
        losses = []

        for i in range(n_batches):
            start = i * self.ttt_batch_size
            end = min(start + self.ttt_batch_size, T - 1)
            if end <= start:
                break

            batch_in = context_ids[:, start:end]
            batch_target = context_ids[:, start + 1:end + 1]

            logits = self.forward(batch_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch_target.reshape(-1))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Re-enable all gradients
        for p in self.parameters():
            p.requires_grad_(True)
        return losses


# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------

def meta_train(model, n_steps=300, seq_len=64, lr=3e-4):
    """Train on repeating-pattern sequences so the model learns structure."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    V = model.vocab_size

    print(f"Meta-training for {n_steps} steps ...")
    for step in range(n_steps):
        # Repeating pattern (learnable structure)
        pat_len = torch.randint(3, 8, (1,)).item()
        pattern = torch.randint(0, V, (pat_len,))
        seq = pattern.repeat(seq_len // pat_len + 2)[:seq_len + 1].unsqueeze(0).to(device)

        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               seq[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1:3d}/{n_steps}, loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("  E2E-TTT Demo (PyTorch — runs on CPU or GPU)")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(42)

    model = E2ETTT(
        vocab_size=64, d_model=64, n_heads=4, n_layers=4,
        d_ff=128, max_seq_len=256, window_size=32,
        ttt_batch_size=16, ttt_lr=0.005,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    ttt_p = sum(p.numel() for p in model.get_ttt_params())
    print(f"  Total params:      {total_p:,}")
    print(f"  TTT params:        {ttt_p:,} ({100 * ttt_p / total_p:.1f}%)")
    print(f"  TTT block indices: {model.ttt_block_indices}")
    print()

    # Meta-train
    t0 = time.time()
    meta_train(model, n_steps=300, seq_len=64, lr=3e-4)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Build structured context
    pattern = torch.tensor([7, 13, 42, 3, 19, 55, 7, 13], device=device)
    context = pattern.repeat(24).unsqueeze(0)  # (1, 192)
    print(f"Test-time training on {context.shape[1]}-token context ...")
    print(f"  Pattern: {pattern.tolist()} (repeating)")

    # Save state
    init_state = model.save_ttt_state()

    # Loss before TTT
    with torch.no_grad():
        logits_b = model(context[:, :-1])
        loss_before = F.cross_entropy(logits_b.reshape(-1, logits_b.size(-1)),
                                      context[:, 1:].reshape(-1)).item()

    # TTT adaptation (3 passes)
    t0 = time.time()
    all_losses = []
    for _ in range(3):
        all_losses.extend(model.ttt_adapt(context))
    ttt_time = time.time() - t0

    # Loss after TTT
    with torch.no_grad():
        logits_a = model(context[:, :-1])
        loss_after = F.cross_entropy(logits_a.reshape(-1, logits_a.size(-1)),
                                     context[:, 1:].reshape(-1)).item()

    print(f"\n  Results:")
    print(f"    Loss before TTT:  {loss_before:.4f}")
    print(f"    Loss after TTT:   {loss_after:.4f}")
    print(f"    Improvement:      {loss_before - loss_after:.4f} ({100 * (loss_before - loss_after) / loss_before:.1f}%)")
    print(f"    TTT time:         {ttt_time:.2f}s")
    print(f"    TTT batch losses: first={all_losses[0]:.4f} -> last={all_losses[-1]:.4f}")

    # Prediction check
    with torch.no_grad():
        test_in = context[:, :8]
        logits = model(test_in)
        predicted = logits.argmax(dim=-1).squeeze(0).tolist()
        expected = context[0, 1:9].tolist()
    print(f"\n  Prediction check:")
    print(f"    Input:     {test_in.squeeze(0).tolist()}")
    print(f"    Expected:  {expected}")
    print(f"    Predicted: {predicted}")
    correct = sum(p == e for p, e in zip(predicted, expected))
    print(f"    Accuracy:  {correct}/{len(expected)}")

    # Restore
    model.restore_ttt_state(init_state)
    with torch.no_grad():
        logits_r = model(context[:, :-1])
        loss_restored = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                        context[:, 1:].reshape(-1)).item()
    print(f"\n  After restoring W_0: loss = {loss_restored:.4f} (matches before: {abs(loss_restored - loss_before) < 1e-4})")
    print()
    print("=" * 60)
    print("  E2E-TTT: compress context into weights, O(1) inference")
    print("  Paper: https://test-time-training.github.io/e2e.pdf")
    print("=" * 60)


if __name__ == "__main__":
    demo()

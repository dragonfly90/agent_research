"""
Simplified E2E-TTT (End-to-End Test-Time Training) Implementation

Based on: "End-to-End Test-Time Training for Long Context"
Paper:   https://test-time-training.github.io/e2e.pdf
arXiv:   https://arxiv.org/abs/2512.23675
Authors: Tandon, Dalal, Li, Koceja, Rod, Buchanan, Wang, Leskovec,
         Koyejo, Hashimoto, Guestrin et al.

This is a simplified PyTorch implementation demonstrating the core ideas:
1. A small Transformer with sliding-window attention
2. Dual MLP blocks (one adapts at test time, one stays frozen)
3. Test-time training via mini-batch gradient descent on next-token prediction
4. Meta-learning the initialization for test-time adaptation

Usage:
    pip install torch
    python ttt_e2e.py
"""

import math
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
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build sliding-window causal mask
        attn_mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            attn_mask[i, start:i + 1] = True
        attn_mask = attn_mask.float().masked_fill(~attn_mask, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale + attn_mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class DualMLP(nn.Module):
    """
    Dual MLP block from E2E-TTT.

    Contains two sub-MLPs:
    - adaptive_mlp: updated during test-time training (stores compressed context)
    - frozen_mlp:   stays frozen to preserve pre-trained knowledge
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # Each sub-MLP gets half the hidden dimension
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
    """Single Transformer block with sliding-window attention + dual MLP."""

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
    """
    Simplified E2E-TTT language model.

    Architecture:
    - Token embedding + learned positional embedding
    - N Transformer blocks with sliding-window attention and dual MLPs
    - Only the adaptive MLPs in the last 25% of blocks are updated at test time

    Test-time training:
    - Process the context in mini-batches of size `ttt_batch_size`
    - For each mini-batch, compute next-token prediction loss and update
      the adaptive MLP parameters via gradient descent
    """

    def __init__(
        self,
        vocab_size=1024,
        d_model=256,
        n_heads=4,
        n_layers=8,
        d_ff=512,
        max_seq_len=2048,
        window_size=256,
        ttt_batch_size=64,
        ttt_lr=1e-4,
    ):
        super().__init__()
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

        # Determine which blocks have TTT-updatable MLPs (last 25%)
        n_ttt = max(1, n_layers // 4)
        self.ttt_block_indices = list(range(n_layers - n_ttt, n_layers))

    def forward(self, idx):
        """Standard forward pass (used during meta-training)."""
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def get_ttt_params(self):
        """Return the adaptive MLP parameters that are updated at test time."""
        params = []
        for i in self.ttt_block_indices:
            params.extend(self.blocks[i].mlp.adaptive_mlp.parameters())
        return params

    def save_ttt_state(self):
        """Save a copy of TTT params (to restore after test-time training)."""
        return {name: p.clone() for name, p in self.named_parameters()
                if any(f"blocks.{i}.mlp.adaptive_mlp" in name
                       for i in self.ttt_block_indices)}

    def restore_ttt_state(self, state):
        """Restore TTT params from saved state."""
        for name, p in self.named_parameters():
            if name in state:
                p.data.copy_(state[name])

    @torch.no_grad()
    def freeze_non_ttt(self):
        """Freeze all parameters except adaptive MLPs in TTT blocks."""
        for name, p in self.named_parameters():
            is_ttt = any(f"blocks.{i}.mlp.adaptive_mlp" in name
                         for i in self.ttt_block_indices)
            p.requires_grad_(is_ttt)

    def ttt_adapt(self, context_ids):
        """
        Test-time training: adapt the model to a long context.

        Processes context_ids in mini-batches. For each mini-batch,
        computes next-token prediction loss and performs a gradient step
        on the adaptive MLP parameters.

        Args:
            context_ids: (1, T) token ids of the context to compress
        """
        self.freeze_non_ttt()
        ttt_params = self.get_ttt_params()
        optimizer = torch.optim.SGD(ttt_params, lr=self.ttt_lr)

        T = context_ids.shape[1]
        n_batches = T // self.ttt_batch_size

        for i in range(n_batches):
            start = i * self.ttt_batch_size
            end = start + self.ttt_batch_size

            # Input and target for next-token prediction
            batch_input = context_ids[:, start:end]
            # Target: shifted by 1 (predict next token)
            if end < T:
                batch_target = context_ids[:, start + 1:end + 1]
            else:
                batch_target = context_ids[:, start + 1:end]
                batch_input = batch_input[:, :batch_target.shape[1]]

            # Forward, loss, backward, step
            logits = self.forward(batch_input)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch_target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Re-enable all gradients for subsequent training
        for p in self.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Meta-training loop (simplified)
# ---------------------------------------------------------------------------

def meta_train(model, n_steps=200, seq_len=512, batch_size=4, lr=3e-4):
    """
    Meta-training: optimize the model initialization for test-time learning.

    For each training step:
    1. Sample a random sequence
    2. Split into context (first half) and query (second half)
    3. Perform TTT on the context (inner loop)
    4. Evaluate loss on the query
    5. Backpropagate through the TTT steps to update W_0 (outer loop)

    This simplified version just trains normally with next-token prediction,
    since full meta-learning requires significant compute.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Meta-training for {n_steps} steps ...")
    for step in range(n_steps):
        # Random token sequence (simulating training data)
        idx = torch.randint(0, 1024, (batch_size, seq_len), device=device)
        target = torch.randint(0, 1024, (batch_size, seq_len), device=device)

        logits = model(idx)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{n_steps}, loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the E2E-TTT workflow."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # 1. Create model
    model = E2ETTT(
        vocab_size=1024,
        d_model=128,
        n_heads=4,
        n_layers=8,
        d_ff=256,
        max_seq_len=1024,
        window_size=64,
        ttt_batch_size=32,
        ttt_lr=1e-2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    ttt_params = sum(p.numel() for p in model.get_ttt_params())
    print(f"Total params:     {total_params:,}")
    print(f"TTT params:       {ttt_params:,} ({100 * ttt_params / total_params:.1f}%)")
    print(f"TTT block indices: {model.ttt_block_indices}")
    print()

    # 2. Meta-train (simplified - just standard training on random data)
    meta_train(model, n_steps=200, seq_len=256, batch_size=4)
    print()

    # 3. Test-time training on a "long context"
    print("Test-time training on a 512-token context ...")
    context = torch.randint(0, 1024, (1, 512), device=device)

    # Save initial state
    init_state = model.save_ttt_state()

    # Measure loss before TTT
    with torch.no_grad():
        logits_before = model(context)
        loss_before = F.cross_entropy(
            logits_before[:, :-1].reshape(-1, logits_before.size(-1)),
            context[:, 1:].reshape(-1),
        )

    # Perform TTT adaptation
    model.ttt_adapt(context)

    # Measure loss after TTT
    with torch.no_grad():
        logits_after = model(context)
        loss_after = F.cross_entropy(
            logits_after[:, :-1].reshape(-1, logits_after.size(-1)),
            context[:, 1:].reshape(-1),
        )

    print(f"  Loss before TTT: {loss_before.item():.4f}")
    print(f"  Loss after TTT:  {loss_after.item():.4f}")
    print(f"  Improvement:     {loss_before.item() - loss_after.item():.4f}")
    print()

    # 4. Restore and show weights are back to original
    model.restore_ttt_state(init_state)
    print("Restored TTT params to initial state (ready for next context).")
    print()
    print("Done! This demonstrates the core E2E-TTT loop:")
    print("  1. Start with meta-trained weights W_0")
    print("  2. For a new long context, run TTT to compress it into weights")
    print("  3. Use adapted weights for prediction")
    print("  4. Restore W_0 for the next context")


if __name__ == "__main__":
    demo()

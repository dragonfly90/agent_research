"""
E2E-TTT (End-to-End Test-Time Training) — JAX/Flax Version

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
    pip install jax jaxlib flax optax
    python3 ttt_e2e_jax.py
"""

import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Sequence


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Multi-head attention with a local sliding window."""
    n_heads: int
    window_size: int

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads

        qkv = nn.Dense(3 * C, use_bias=False, name="qkv")(x)  # (B, T, 3C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = jnp.transpose(q, (0, 2, 1, 3))  # (B, n_heads, T, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = head_dim ** -0.5
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

        # Sliding-window causal mask
        idx = jnp.arange(T)
        mask = (idx[:, None] - idx[None, :] >= 0) & (idx[:, None] - idx[None, :] < self.window_size)
        attn = jnp.where(mask[None, None, :, :], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)
        return nn.Dense(C, use_bias=False, name="out_proj")(out)


class DualMLP(nn.Module):
    """
    Dual MLP from E2E-TTT.
    - adaptive: updated during TTT
    - frozen:   stays fixed
    """
    d_ff: int

    @nn.compact
    def __call__(self, x):
        half_ff = self.d_ff // 2
        # Adaptive MLP
        a = nn.Dense(half_ff, use_bias=False, name="adaptive_fc1")(x)
        a = nn.gelu(a)
        a = nn.Dense(x.shape[-1], use_bias=False, name="adaptive_fc2")(a)
        # Frozen MLP
        f = nn.Dense(half_ff, use_bias=False, name="frozen_fc1")(x)
        f = nn.gelu(f)
        f = nn.Dense(x.shape[-1], use_bias=False, name="frozen_fc2")(f)
        return a + f


class TransformerBlock(nn.Module):
    n_heads: int
    d_ff: int
    window_size: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(name="ln1")(x)
        x = x + SlidingWindowAttention(self.n_heads, self.window_size, name="attn")(h)
        h = nn.LayerNorm(name="ln2")(x)
        x = x + DualMLP(self.d_ff, name="mlp")(h)
        return x


# ---------------------------------------------------------------------------
# E2E-TTT Model
# ---------------------------------------------------------------------------

class E2ETTTModel(nn.Module):
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 128
    max_seq_len: int = 256
    window_size: int = 32

    @nn.compact
    def __call__(self, idx):
        B, T = idx.shape
        tok = nn.Embed(self.vocab_size, self.d_model, name="tok_emb")(idx)
        pos = nn.Embed(self.max_seq_len, self.d_model, name="pos_emb")(jnp.arange(T))
        x = tok + pos
        for i in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.d_ff, self.window_size,
                                 name=f"block_{i}")(x)
        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(self.vocab_size, use_bias=False, name="head")(x)
        return logits


# ---------------------------------------------------------------------------
# TTT utilities
# ---------------------------------------------------------------------------

def get_ttt_block_indices(n_layers):
    """Last 25% of blocks."""
    n_ttt = max(1, n_layers // 4)
    return list(range(n_layers - n_ttt, n_layers))


def split_params(params, ttt_indices):
    """Split params into ttt (adaptive MLPs in TTT blocks) and frozen."""
    from flax.core import freeze, unfreeze
    params = unfreeze(params)
    ttt_params = {}
    frozen_params = {}

    for key, val in params['params'].items():
        if key.startswith('block_'):
            block_idx = int(key.split('_')[1])
            if block_idx in ttt_indices:
                # Split this block: adaptive MLP -> ttt, rest -> frozen
                ttt_block = {}
                frozen_block = {}
                for subkey, subval in val.items():
                    if subkey == 'mlp':
                        ttt_mlp = {}
                        frozen_mlp = {}
                        for mlp_key, mlp_val in subval.items():
                            if mlp_key.startswith('adaptive'):
                                ttt_mlp[mlp_key] = mlp_val
                            else:
                                frozen_mlp[mlp_key] = mlp_val
                        if ttt_mlp:
                            ttt_block['mlp'] = ttt_mlp
                        if frozen_mlp:
                            frozen_block['mlp'] = frozen_mlp
                    else:
                        frozen_block[subkey] = subval
                if ttt_block:
                    ttt_params[key] = ttt_block
                if frozen_block:
                    frozen_params[key] = frozen_block
            else:
                frozen_params[key] = val
        else:
            frozen_params[key] = val

    return freeze({'params': ttt_params}), freeze({'params': frozen_params})


def merge_params(ttt_params, frozen_params):
    """Merge TTT and frozen params back together."""
    from flax.core import freeze, unfreeze
    ttt = unfreeze(ttt_params)['params']
    frozen = unfreeze(frozen_params)['params']
    merged = {}

    all_keys = set(list(ttt.keys()) + list(frozen.keys()))
    for key in sorted(all_keys):
        if key in ttt and key in frozen:
            merged[key] = {}
            all_subkeys = set(list(ttt[key].keys()) + list(frozen[key].keys()))
            for subkey in all_subkeys:
                if subkey in ttt[key] and subkey in frozen[key]:
                    merged[key][subkey] = {**ttt[key][subkey], **frozen[key][subkey]}
                elif subkey in ttt[key]:
                    merged[key][subkey] = ttt[key][subkey]
                else:
                    merged[key][subkey] = frozen[key][subkey]
        elif key in ttt:
            merged[key] = ttt[key]
        else:
            merged[key] = frozen[key]

    return freeze({'params': merged})


def ttt_loss_fn(ttt_params, frozen_params, model, batch_in, batch_target):
    """Loss function for TTT: next-token prediction."""
    params = merge_params(ttt_params, frozen_params)
    logits = model.apply(params, batch_in)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        batch_target.reshape(-1)
    ).mean()
    return loss


def ttt_adapt(model, params, context, ttt_indices, ttt_batch_size=16,
              ttt_lr=0.005, n_passes=3):
    """
    Test-time training: adapt adaptive MLP params to a context.
    Returns updated params and list of losses.
    """
    ttt_params, frozen_params = split_params(params, ttt_indices)
    optimizer = optax.sgd(ttt_lr)
    opt_state = optimizer.init(ttt_params)

    grad_fn = jax.grad(ttt_loss_fn, argnums=0)
    T = context.shape[1]
    losses = []

    for _ in range(n_passes):
        n_batches = max(1, T // ttt_batch_size)
        for i in range(n_batches):
            start = i * ttt_batch_size
            end = min(start + ttt_batch_size, T - 1)
            if end <= start:
                break
            batch_in = context[:, start:end]
            batch_target = context[:, start + 1:end + 1]

            loss = ttt_loss_fn(ttt_params, frozen_params, model, batch_in, batch_target)
            losses.append(float(loss))

            grads = grad_fn(ttt_params, frozen_params, model, batch_in, batch_target)
            updates, opt_state = optimizer.update(grads, opt_state)
            ttt_params = optax.apply_updates(ttt_params, updates)

    updated_params = merge_params(ttt_params, frozen_params)
    return updated_params, losses


# ---------------------------------------------------------------------------
# Meta-training
# ---------------------------------------------------------------------------

def create_train_state(rng, model, lr=3e-4):
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(rng, dummy)
    tx = optax.adamw(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, inputs, targets):
    def loss_fn(params):
        logits = state.apply_fn(params, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        ).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def meta_train(state, model, n_steps=300, seq_len=64):
    V = model.vocab_size
    rng = jax.random.PRNGKey(0)

    print(f"Meta-training for {n_steps} steps ...")
    for step in range(n_steps):
        rng, key1, key2 = jax.random.split(rng, 3)
        pat_len = jax.random.randint(key1, (), 3, 8).item()
        pattern = jax.random.randint(key2, (pat_len,), 0, V)
        repeats = seq_len // pat_len + 2
        seq = jnp.tile(pattern, repeats)[:seq_len + 1]
        inputs = seq[:-1][None, :]  # (1, seq_len)
        targets = seq[1:][None, :]

        state, loss = train_step(state, inputs, targets)

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1:3d}/{n_steps}, loss: {float(loss):.4f}")

    return state


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("  E2E-TTT Demo (JAX/Flax)")
    print("=" * 60)
    print()
    print(f"  JAX backend: {jax.default_backend()}")
    print()

    model = E2ETTTModel(
        vocab_size=64, d_model=64, n_heads=4, n_layers=4,
        d_ff=128, max_seq_len=256, window_size=32,
    )

    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, lr=3e-4)

    total_p = sum(x.size for x in jax.tree.leaves(state.params))
    ttt_indices = get_ttt_block_indices(model.n_layers)
    ttt_p_tree, _ = split_params(state.params, ttt_indices)
    ttt_p = sum(x.size for x in jax.tree.leaves(ttt_p_tree))
    print(f"  Total params:      {total_p:,}")
    print(f"  TTT params:        {ttt_p:,} ({100 * ttt_p / total_p:.1f}%)")
    print(f"  TTT block indices: {ttt_indices}")
    print()

    # Meta-train
    t0 = time.time()
    state = meta_train(state, model, n_steps=300, seq_len=64)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Build context
    pattern = jnp.array([7, 13, 42, 3, 19, 55, 7, 13])
    context = jnp.tile(pattern, 24)[None, :]  # (1, 192)
    print(f"Test-time training on {context.shape[1]}-token context ...")
    print(f"  Pattern: {pattern.tolist()} (repeating)")

    # Loss before TTT
    logits_b = model.apply(state.params, context[:, :-1])
    loss_before = float(optax.softmax_cross_entropy_with_integer_labels(
        logits_b.reshape(-1, logits_b.shape[-1]),
        context[:, 1:].reshape(-1)
    ).mean())

    # TTT adapt
    t0 = time.time()
    adapted_params, ttt_losses = ttt_adapt(
        model, state.params, context, ttt_indices,
        ttt_batch_size=16, ttt_lr=0.005, n_passes=3,
    )
    ttt_time = time.time() - t0

    # Loss after TTT
    logits_a = model.apply(adapted_params, context[:, :-1])
    loss_after = float(optax.softmax_cross_entropy_with_integer_labels(
        logits_a.reshape(-1, logits_a.shape[-1]),
        context[:, 1:].reshape(-1)
    ).mean())

    print(f"\n  Results:")
    print(f"    Loss before TTT:  {loss_before:.4f}")
    print(f"    Loss after TTT:   {loss_after:.4f}")
    improvement = loss_before - loss_after
    print(f"    Improvement:      {improvement:.4f} ({100 * improvement / loss_before:.1f}%)")
    print(f"    TTT time:         {ttt_time:.2f}s")
    print(f"    TTT batch losses: first={ttt_losses[0]:.4f} -> last={ttt_losses[-1]:.4f}")

    # Prediction check
    test_in = context[:, :8]
    logits = model.apply(adapted_params, test_in)
    predicted = jnp.argmax(logits, axis=-1).squeeze(0).tolist()
    expected = context[0, 1:9].tolist()
    print(f"\n  Prediction check:")
    print(f"    Input:     {test_in.squeeze(0).tolist()}")
    print(f"    Expected:  {expected}")
    print(f"    Predicted: {predicted}")
    correct = sum(p == e for p, e in zip(predicted, expected))
    print(f"    Accuracy:  {correct}/{len(expected)}")

    # Restore check
    logits_r = model.apply(state.params, context[:, :-1])
    loss_restored = float(optax.softmax_cross_entropy_with_integer_labels(
        logits_r.reshape(-1, logits_r.shape[-1]),
        context[:, 1:].reshape(-1)
    ).mean())
    print(f"\n  Original W_0 loss:   {loss_restored:.4f} (matches before: {abs(loss_restored - loss_before) < 1e-4})")
    print()
    print("=" * 60)
    print("  E2E-TTT: compress context into weights, O(1) inference")
    print("  Paper: https://test-time-training.github.io/e2e.pdf")
    print("=" * 60)


if __name__ == "__main__":
    demo()

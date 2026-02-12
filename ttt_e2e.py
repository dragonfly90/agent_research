"""
Simplified E2E-TTT (End-to-End Test-Time Training) — Pure NumPy

Based on: "End-to-End Test-Time Training for Long Context"
Paper:   https://test-time-training.github.io/e2e.pdf
arXiv:   https://arxiv.org/abs/2512.23675
Authors: Tandon, Dalal, Li, Koceja, Rod, Buchanan, Wang, Leskovec,
         Koyejo, Hashimoto, Guestrin et al.

Core ideas demonstrated (no GPU / no PyTorch required):
1. Sliding-window causal attention
2. Dual MLP blocks (one adapts at test time, one stays frozen)
3. Test-time training via mini-batch gradient descent on next-token prediction
4. Meta-training the initialization

Usage:
    python3 ttt_e2e.py
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(x):
    """Derivative of GELU w.r.t. x."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    inner_grad = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_grad


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(logits, targets):
    """Cross-entropy loss. logits: (T, V), targets: (T,) integer indices."""
    T, V = logits.shape
    probs = softmax(logits, axis=-1)
    log_probs = np.log(probs + 1e-12)
    loss = -np.mean(log_probs[np.arange(T), targets])
    # Gradient w.r.t. logits
    grad = probs.copy()
    grad[np.arange(T), targets] -= 1.0
    grad /= T
    return loss, grad


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta, x_norm, mean, var


# ---------------------------------------------------------------------------
# Layer classes with forward + backward
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.grad_W = np.zeros_like(self.W)

    def forward(self, x):
        self.x = x
        return x @ self.W

    def backward(self, grad_out):
        # x: (..., in_dim), grad_out: (..., out_dim)
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        g_flat = grad_out.reshape(-1, grad_out.shape[-1])
        self.grad_W = x_flat.T @ g_flat
        grad_in = grad_out @ self.W.T
        return grad_in


class LayerNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def forward(self, x, eps=1e-5):
        self.x = x
        self.eps = eps
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_out):
        N = self.x.shape[-1]
        dx_norm = grad_out * self.gamma
        self.grad_gamma = np.sum(grad_out * self.x_norm, axis=tuple(range(grad_out.ndim - 1)))
        self.grad_beta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        dx = (1.0 / N) * std_inv * (N * dx_norm - np.sum(dx_norm, axis=-1, keepdims=True)
              - self.x_norm * np.sum(dx_norm * self.x_norm, axis=-1, keepdims=True))
        return dx


class MLPBlock:
    """Two-layer MLP with GELU activation."""

    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        self.h_pre = self.fc1.forward(x)
        self.h = gelu(self.h_pre)
        return self.fc2.forward(self.h)

    def backward(self, grad_out):
        grad_h = self.fc2.backward(grad_out)
        grad_h_pre = grad_h * gelu_backward(self.h_pre)
        return self.fc1.backward(grad_h_pre)

    def params_and_grads(self):
        return [(self.fc1.W, self.fc1.grad_W), (self.fc2.W, self.fc2.grad_W)]


# ---------------------------------------------------------------------------
# Sliding-Window Attention
# ---------------------------------------------------------------------------

class SlidingWindowAttention:
    def __init__(self, d_model, n_heads, window_size):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.Wqkv = Linear(d_model, 3 * d_model)
        self.Wout = Linear(d_model, d_model)

    def forward(self, x):
        T, C = x.shape
        qkv = self.Wqkv.forward(x)  # (T, 3*C)
        qkv = qkv.reshape(T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each: (T, n_heads, head_dim)
        q = q.transpose(1, 0, 2)  # (n_heads, T, head_dim)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # (n_heads, T, T)

        # Sliding-window causal mask
        mask = np.full((T, T), -1e9, dtype=np.float32)
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0.0
        scores = scores + mask

        self.attn_weights = softmax(scores, axis=-1)  # (n_heads, T, T)
        out = np.matmul(self.attn_weights, v)  # (n_heads, T, head_dim)
        out = out.transpose(1, 0, 2).reshape(T, C)  # (T, C)
        return self.Wout.forward(out)

    def backward(self, grad_out):
        # Simplified: backprop through Wout, then project back through Wqkv
        grad_mid = self.Wout.backward(grad_out)  # (T, d_model)
        # Wqkv maps (T, d_model) -> (T, 3*d_model), so we need to expand
        # grad_mid to (T, 3*d_model) for the backward (gradient flows through q,k,v)
        T, C = grad_mid.shape
        grad_qkv = np.concatenate([grad_mid, grad_mid, grad_mid], axis=-1)  # approximate
        self.Wqkv.x = self.Wqkv.x  # already cached from forward
        x_flat = self.Wqkv.x.reshape(-1, self.Wqkv.x.shape[-1])
        g_flat = grad_qkv.reshape(-1, grad_qkv.shape[-1])
        self.Wqkv.grad_W = x_flat.T @ g_flat
        grad_in = grad_qkv @ self.Wqkv.W.T
        return grad_in

    def params_and_grads(self):
        return [(self.Wqkv.W, self.Wqkv.grad_W), (self.Wout.W, self.Wout.grad_W)]


# ---------------------------------------------------------------------------
# Transformer Block with Dual MLP
# ---------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, window_size):
        self.ln1 = LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.ln2 = LayerNorm(d_model)
        # Dual MLP: adaptive (updated at test time) + frozen
        self.adaptive_mlp = MLPBlock(d_model, d_ff // 2)
        self.frozen_mlp = MLPBlock(d_model, d_ff // 2)

    def forward(self, x):
        # Attention sub-layer
        self.x1 = x
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h)
        # MLP sub-layer (dual)
        self.x2 = x
        h2 = self.ln2.forward(x)
        x = x + self.adaptive_mlp.forward(h2) + self.frozen_mlp.forward(h2)
        return x

    def backward(self, grad_out):
        # MLP backward
        grad_adaptive = self.adaptive_mlp.backward(grad_out)
        grad_frozen = self.frozen_mlp.backward(grad_out)
        grad_ln2 = self.ln2.backward(grad_adaptive + grad_frozen)
        grad_x2 = grad_out + grad_ln2
        # Attention backward
        grad_attn = self.attn.backward(grad_x2)
        grad_ln1 = self.ln1.backward(grad_attn)
        return grad_x2 + grad_ln1

    def all_params_and_grads(self):
        pgs = []
        pgs.extend(self.attn.params_and_grads())
        pgs.extend(self.adaptive_mlp.params_and_grads())
        pgs.extend(self.frozen_mlp.params_and_grads())
        pgs.append((self.ln1.gamma, self.ln1.grad_gamma))
        pgs.append((self.ln1.beta, self.ln1.grad_beta))
        pgs.append((self.ln2.gamma, self.ln2.grad_gamma))
        pgs.append((self.ln2.beta, self.ln2.grad_beta))
        return pgs

    def adaptive_params_and_grads(self):
        """Only the adaptive MLP params (for TTT updates)."""
        return self.adaptive_mlp.params_and_grads()


# ---------------------------------------------------------------------------
# E2E-TTT Model
# ---------------------------------------------------------------------------

class E2ETTT:
    """
    Simplified E2E-TTT language model (pure numpy).

    Architecture:
    - Token embedding + learned positional embedding
    - N Transformer blocks with sliding-window attention and dual MLPs
    - Only adaptive MLPs in the last 25% of blocks are updated at test time
    """

    def __init__(self, vocab_size=256, d_model=64, n_heads=4, n_layers=4,
                 d_ff=128, max_seq_len=512, window_size=32,
                 ttt_batch_size=16, ttt_lr=0.01):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.ttt_batch_size = ttt_batch_size
        self.ttt_lr = ttt_lr

        # Embeddings
        self.tok_emb = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
        self.pos_emb = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, window_size)
                       for _ in range(n_layers)]

        # Final layer norm + output head
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

        # TTT block indices: last 25%
        n_ttt = max(1, n_layers // 4)
        self.ttt_block_indices = list(range(n_layers - n_ttt, n_layers))

    def forward(self, token_ids):
        """Forward pass. token_ids: (T,) integer array. Returns logits (T, V)."""
        T = len(token_ids)
        x = self.tok_emb[token_ids] + self.pos_emb[:T]  # (T, d_model)
        self.block_inputs = [x]
        for block in self.blocks:
            x = block.forward(x)
            self.block_inputs.append(x)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)  # (T, vocab_size)
        return logits

    def backward(self, grad_logits):
        """Backward pass through the whole model."""
        grad = self.head.backward(grad_logits)
        grad = self.ln_f.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)

    def all_params_and_grads(self):
        """All trainable parameters and their gradients."""
        pgs = []
        for block in self.blocks:
            pgs.extend(block.all_params_and_grads())
        pgs.append((self.ln_f.gamma, self.ln_f.grad_gamma))
        pgs.append((self.ln_f.beta, self.ln_f.grad_beta))
        pgs.append((self.head.W, self.head.grad_W))
        return pgs

    def ttt_params_and_grads(self):
        """Only adaptive MLP params in TTT blocks."""
        pgs = []
        for i in self.ttt_block_indices:
            pgs.extend(self.blocks[i].adaptive_params_and_grads())
        return pgs

    def save_ttt_state(self):
        """Snapshot adaptive MLP weights."""
        state = []
        for i in self.ttt_block_indices:
            for p, _ in self.blocks[i].adaptive_params_and_grads():
                state.append(p.copy())
        return state

    def restore_ttt_state(self, state):
        """Restore adaptive MLP weights from snapshot."""
        idx = 0
        for i in self.ttt_block_indices:
            for p, _ in self.blocks[i].adaptive_params_and_grads():
                p[:] = state[idx]
                idx += 1

    def count_params(self):
        total = self.tok_emb.size + self.pos_emb.size
        for block in self.blocks:
            for p, _ in block.all_params_and_grads():
                total += p.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size + self.head.W.size
        return total

    def count_ttt_params(self):
        total = 0
        for p, _ in self.ttt_params_and_grads():
            total += p.size
        return total

    def ttt_adapt(self, context_ids):
        """
        Test-time training: adapt the model to a long context.

        Process context in mini-batches, compute next-token prediction loss,
        update only the adaptive MLP parameters via SGD.
        """
        T = len(context_ids)
        n_batches = max(1, T // self.ttt_batch_size)
        losses = []

        for i in range(n_batches):
            start = i * self.ttt_batch_size
            end = min(start + self.ttt_batch_size, T - 1)
            if end <= start:
                break

            batch_in = context_ids[start:end]
            batch_target = context_ids[start + 1:end + 1]

            logits = self.forward(batch_in)
            loss, grad_logits = cross_entropy(logits, batch_target)
            losses.append(loss)

            self.backward(grad_logits)

            # SGD step on TTT params only
            for p, g in self.ttt_params_and_grads():
                p -= self.ttt_lr * g

        return losses


# ---------------------------------------------------------------------------
# Meta-training (simplified)
# ---------------------------------------------------------------------------

def meta_train(model, n_steps=100, seq_len=64, lr=1e-3):
    """
    Simplified meta-training via standard next-token prediction.

    Uses a synthetic dataset with repeating patterns so the model
    can learn something meaningful (unlike pure random tokens).
    """
    print(f"Meta-training for {n_steps} steps (seq_len={seq_len}) ...")
    V = model.vocab_size

    for step in range(n_steps):
        # Generate a sequence with repeating patterns (learnable structure)
        pattern_len = np.random.randint(3, 8)
        pattern = np.random.randint(0, V, size=pattern_len)
        repeats = seq_len // pattern_len + 1
        seq = np.tile(pattern, repeats)[:seq_len + 1].astype(np.int32)

        inputs = seq[:-1]
        targets = seq[1:]

        logits = model.forward(inputs)
        loss, grad_logits = cross_entropy(logits, targets)
        model.backward(grad_logits)

        # SGD on all params
        for p, g in model.all_params_and_grads():
            p -= lr * g

        if (step + 1) % 25 == 0:
            print(f"  Step {step + 1:3d}/{n_steps}, loss: {loss:.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=" * 60)
    print("  E2E-TTT Demo (Pure NumPy — no GPU required)")
    print("=" * 60)
    print()

    np.random.seed(42)

    # 1. Create model (small for CPU)
    model = E2ETTT(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=4,
        d_ff=64,
        max_seq_len=256,
        window_size=16,
        ttt_batch_size=16,
        ttt_lr=0.005,
    )

    total_p = model.count_params()
    ttt_p = model.count_ttt_params()
    print(f"Model config:")
    print(f"  vocab_size=64, d_model=32, n_layers=4, n_heads=4")
    print(f"  window_size=16, ttt_batch_size=16")
    print(f"  Total params:      {total_p:,}")
    print(f"  TTT params:        {ttt_p:,} ({100 * ttt_p / total_p:.1f}%)")
    print(f"  TTT block indices: {model.ttt_block_indices}")
    print()

    # 2. Meta-train on repeating-pattern sequences
    t0 = time.time()
    meta_train(model, n_steps=300, seq_len=64, lr=1e-3)
    print(f"  Meta-training time: {time.time() - t0:.1f}s")
    print()

    # 3. Test-time training on a structured context
    #    Create a context with a repeating pattern the model hasn't seen
    pattern = np.array([7, 13, 42, 3, 19, 55, 7, 13], dtype=np.int32)
    context = np.tile(pattern, 24)  # 192 tokens — longer context for more TTT steps
    print(f"Test-time training on a {len(context)}-token context ...")
    print(f"  Context pattern: {pattern.tolist()} (repeating)")

    # Save initial state
    init_state = model.save_ttt_state()

    # Measure loss BEFORE TTT
    logits_before = model.forward(context[:-1])
    loss_before, _ = cross_entropy(logits_before, context[1:])

    # Perform TTT adaptation (multiple passes for deeper compression)
    t0 = time.time()
    ttt_losses = []
    for ttt_pass in range(3):
        ttt_losses.extend(model.ttt_adapt(context))
    ttt_time = time.time() - t0

    # Measure loss AFTER TTT
    logits_after = model.forward(context[:-1])
    loss_after, _ = cross_entropy(logits_after, context[1:])

    print()
    print(f"  Results:")
    print(f"    Loss before TTT:  {loss_before:.4f}")
    print(f"    Loss after TTT:   {loss_after:.4f}")
    print(f"    Improvement:      {loss_before - loss_after:.4f} ({100 * (loss_before - loss_after) / loss_before:.1f}%)")
    print(f"    TTT time:         {ttt_time:.2f}s")
    print(f"    TTT batch losses: {['%.4f' % l for l in ttt_losses]}")
    print()

    # 4. Show prediction quality
    test_input = context[:8]  # Feed start of pattern
    logits = model.forward(test_input)
    predicted = np.argmax(logits, axis=-1)
    expected = context[1:9]
    print(f"  Prediction check (after TTT):")
    print(f"    Input:     {test_input.tolist()}")
    print(f"    Expected:  {expected.tolist()}")
    print(f"    Predicted: {predicted.tolist()}")
    correct = np.sum(predicted == expected)
    print(f"    Accuracy:  {correct}/{len(expected)}")
    print()

    # 5. Restore original weights
    model.restore_ttt_state(init_state)
    logits_restored = model.forward(context[:-1])
    loss_restored, _ = cross_entropy(logits_restored, context[1:])
    print(f"  After restoring W_0: loss = {loss_restored:.4f} (matches before: {abs(loss_restored - loss_before) < 1e-5})")
    print()

    print("=" * 60)
    print("  Summary: E2E-TTT Workflow")
    print("=" * 60)
    print("  1. Meta-train model on diverse data -> get good W_0")
    print("  2. At test time, receive long context")
    print("  3. Run TTT: mini-batch SGD on next-token loss")
    print("     -> compresses context into adaptive MLP weights")
    print("  4. Use adapted model for prediction (constant latency)")
    print("  5. Restore W_0 for next context")
    print()
    print("  Key advantage: O(1) inference latency vs O(n^2) for full attention")
    print("  Paper: https://test-time-training.github.io/e2e.pdf")


if __name__ == "__main__":
    demo()

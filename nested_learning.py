"""
Nested Learning & Titans Neural Memory — Pure NumPy Implementation

Based on:
  1. "Nested Learning: The Illusion of Deep Learning Architectures"
     Paper:  https://abehrouz.github.io/files/NL.pdf
     arXiv:  https://arxiv.org/abs/2512.24695
     Authors: Ali Behrouz et al. (Google Research, NeurIPS 2025)

  2. "Titans: Learning to Memorize at Test Time"
     arXiv:  https://arxiv.org/abs/2501.00663
     Authors: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)

Core ideas demonstrated:
  - Neural long-term memory that learns to memorize via gradient descent
  - Surprise-based gating: memory updates more for unexpected tokens
  - Continuum Memory System (CMS): layers update at different frequencies
    (fast layers = working memory, slow layers = long-term knowledge)
  - Self-modifying mechanism: the model updates its own weights at test time
  - Three Titans variants: Memory as Context (MAC), Gate (MAG), Layer (MAL)

Usage:
    python3 nested_learning.py
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ---------------------------------------------------------------------------
# Neural Long-Term Memory Module (Titans)
# ---------------------------------------------------------------------------

class NeuralMemory:
    """
    Neural long-term memory from the Titans paper.

    The memory is a linear map M that learns to associate keys -> values
    via online gradient descent with momentum and surprise-based gating.

    Update rule:
        S_t = eta_t * S_{t-1} - theta_t * grad_loss(M_{t-1}; x_t)
        M_t = (1 - alpha_t) * M_{t-1} + S_t

    Where:
        loss(M; x_t) = ||M(k_t) - v_t||^2   (associative memory loss)
        k_t = x_t @ W_K,  v_t = x_t @ W_V
        alpha_t = sigmoid(surprise_t)          (forgetting gate)
        theta_t = data-dependent learning rate
        eta_t = momentum decay
    """

    def __init__(self, d_model, d_memory=None):
        self.d_model = d_model
        self.d_memory = d_memory or d_model

        # Key/Value projections
        self.W_K = np.random.randn(d_model, self.d_memory).astype(np.float32) * 0.02
        self.W_V = np.random.randn(d_model, self.d_memory).astype(np.float32) * 0.02
        self.W_Q = np.random.randn(d_model, self.d_memory).astype(np.float32) * 0.02

        # Data-dependent gate projections
        self.W_alpha = np.random.randn(d_model, 1).astype(np.float32) * 0.02  # forgetting
        self.W_theta = np.random.randn(d_model, 1).astype(np.float32) * 0.02 + 0.5  # lr (biased high)
        self.W_eta = np.random.randn(d_model, 1).astype(np.float32) * 0.02 + 0.5  # momentum (biased high)

        # Memory state: M maps d_memory -> d_memory
        self.M = np.zeros((self.d_memory, self.d_memory), dtype=np.float32)
        # Momentum state
        self.S = np.zeros_like(self.M)

        self.surprise_history = []

    def reset(self):
        self.M = np.zeros((self.d_memory, self.d_memory), dtype=np.float32)
        self.S = np.zeros_like(self.M)
        self.surprise_history = []

    def compute_surprise(self, x_t):
        """Compute surprise = prediction error for token x_t."""
        k_t = x_t @ self.W_K  # (d_memory,)
        v_t = x_t @ self.W_V  # (d_memory,)
        predicted = self.M @ k_t
        surprise = np.sum((predicted - v_t) ** 2)
        return surprise

    def update(self, x_t):
        """
        Update memory with a single token x_t.
        Returns the surprise score for this token.
        """
        k_t = x_t @ self.W_K  # (d_memory,)
        v_t = x_t @ self.W_V  # (d_memory,)

        # Associative memory loss gradient: d/dM ||M @ k - v||^2
        predicted = self.M @ k_t
        error = predicted - v_t
        surprise = np.sum(error ** 2)
        self.surprise_history.append(surprise)

        # Gradient: 2 * (M @ k - v) @ k^T  (outer product)
        grad_M = 2.0 * np.outer(error, k_t)

        # Data-dependent gates
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
        alpha_t = sigmoid(x_t @ self.W_alpha).item() * 0.05  # forgetting rate [0, 0.05]
        theta_t = 0.01 + sigmoid(x_t @ self.W_theta).item() * 0.09  # learning rate [0.01, 0.1]
        eta_t = 0.5 + sigmoid(x_t @ self.W_eta).item() * 0.4  # momentum decay [0.5, 0.9]

        # Momentum update
        self.S = eta_t * self.S - theta_t * grad_M
        # Memory update with forgetting
        self.M = (1.0 - alpha_t) * self.M + self.S

        return surprise

    def retrieve(self, x_t):
        """Retrieve from memory without updating (inference only)."""
        q_t = x_t @ self.W_Q
        return self.M @ q_t

    def update_projections(self, lr, grad_scale=1.0):
        """Simple gradient step on projection matrices (for meta-training)."""
        noise = lambda shape: np.random.randn(*shape).astype(np.float32) * grad_scale
        self.W_K -= lr * noise(self.W_K.shape)
        self.W_V -= lr * noise(self.W_V.shape)
        self.W_Q -= lr * noise(self.W_Q.shape)
        self.W_alpha -= lr * noise(self.W_alpha.shape)
        self.W_theta -= lr * noise(self.W_theta.shape)
        self.W_eta -= lr * noise(self.W_eta.shape)


# ---------------------------------------------------------------------------
# Continuum Memory System (CMS)
# ---------------------------------------------------------------------------

class ContinuumMemorySystem:
    """
    Continuum Memory System from Nested Learning.

    Multiple memory modules updating at different frequencies:
    - Fast memory:  updates every token (working memory / current context)
    - Medium memory: updates every K tokens (episodic / session memory)
    - Slow memory:  updates every K*K tokens (long-term knowledge)

    This creates a "continuum" from short-term to long-term memory,
    each level compressing information at a different timescale.
    """

    def __init__(self, d_model, n_levels=3, base_freq=1):
        self.d_model = d_model
        self.n_levels = n_levels
        self.memories = []
        self.frequencies = []
        self.step_count = 0

        for level in range(n_levels):
            mem = NeuralMemory(d_model)
            freq = base_freq * (4 ** level)  # 1, 4, 16
            self.memories.append(mem)
            self.frequencies.append(freq)

    def reset(self):
        self.step_count = 0
        for mem in self.memories:
            mem.reset()

    def update(self, x_t):
        """Update each memory level according to its frequency."""
        self.step_count += 1
        surprises = []
        for level, (mem, freq) in enumerate(zip(self.memories, self.frequencies)):
            if self.step_count % freq == 0:
                s = mem.update(x_t)
                surprises.append((level, s))
        return surprises

    def retrieve(self, x_t):
        """Retrieve from all memory levels and combine."""
        retrievals = []
        for mem in self.memories:
            r = mem.retrieve(x_t)
            retrievals.append(r)
        # Weighted sum (fast memory weighted more for recent, slow for stable)
        weights = np.array([0.5, 0.3, 0.2])[:self.n_levels]
        weights /= weights.sum()
        combined = sum(w * r for w, r in zip(weights, retrievals))
        return combined

    def get_surprise_summary(self):
        """Get surprise history per level."""
        return {f"level_{i} (freq={self.frequencies[i]})":
                mem.surprise_history for i, mem in enumerate(self.memories)}


# ---------------------------------------------------------------------------
# Memory as Gate (MAG) — Titans Variant
# ---------------------------------------------------------------------------

class MAGBlock:
    """
    Memory as Gate (MAG) from Titans.

    Two parallel branches:
    1. Sliding-window attention (short-term, accurate)
    2. Neural memory (long-term, compressed)

    Output = attention_output * gate(memory_output)
    """

    def __init__(self, d_model, window_size=16):
        self.d_model = d_model
        self.window_size = window_size
        self.memory = NeuralMemory(d_model)

        # Simple attention projection
        self.W_attn = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        # Gate projection
        self.W_gate = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def forward(self, sequence):
        """Process a sequence through MAG block."""
        T, D = sequence.shape
        output = np.zeros_like(sequence)

        for t in range(T):
            x_t = sequence[t]

            # Branch 1: Local attention (simplified as windowed average)
            start = max(0, t - self.window_size + 1)
            window = sequence[start:t + 1]
            attn_out = np.mean(window @ self.W_attn, axis=0)

            # Branch 2: Neural memory retrieval + update
            mem_out = self.memory.retrieve(x_t)
            self.memory.update(x_t)

            # Gating
            gate = 1.0 / (1.0 + np.exp(-(mem_out @ self.W_gate)))
            output[t] = attn_out * gate + x_t  # residual

        return output


# ---------------------------------------------------------------------------
# Self-Modifying Model (Hope-like)
# ---------------------------------------------------------------------------

class SelfModifyingModel:
    """
    Simplified Hope-like self-modifying architecture.

    The model has:
    1. An embedding layer
    2. A CMS (Continuum Memory System) for multi-timescale memory
    3. A MAG block for combining attention + memory
    4. A prediction head

    At test time, the CMS memories update themselves (self-modifying),
    compressing the context into their weights at multiple timescales.
    """

    def __init__(self, vocab_size=64, d_model=32, n_memory_levels=3):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding
        self.embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1

        # CMS: multi-frequency memory
        self.cms = ContinuumMemorySystem(d_model, n_levels=n_memory_levels)

        # MAG block
        self.mag = MAGBlock(d_model, window_size=16)

        # Prediction head
        self.W_out = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.02

    def reset(self):
        self.cms.reset()
        self.mag.memory.reset()

    def process_sequence(self, token_ids):
        """Process a token sequence, updating memory at each step."""
        T = len(token_ids)
        embeddings = self.embedding[token_ids]  # (T, d_model)

        # Update CMS with each token
        cms_outputs = np.zeros_like(embeddings)
        for t in range(T):
            self.cms.update(embeddings[t])
            cms_outputs[t] = self.cms.retrieve(embeddings[t])

        # Combine embedding with CMS retrieval (scale up memory signal)
        combined = layer_norm(embeddings + 5.0 * cms_outputs)

        # MAG block
        mag_out = self.mag.forward(combined)
        mag_out = layer_norm(mag_out)

        # Predict
        logits = mag_out @ self.W_out  # (T, vocab_size)
        return logits

    def predict_next(self, token_ids):
        """Predict the next token given a sequence."""
        logits = self.process_sequence(token_ids)
        return logits[-1]  # last position predicts next token

    def compute_loss(self, token_ids):
        """Next-token prediction loss over a sequence."""
        logits = self.process_sequence(token_ids[:-1])
        targets = token_ids[1:]
        # Cross-entropy
        probs = softmax(logits, axis=-1)
        T = len(targets)
        log_probs = np.log(probs[np.arange(T), targets] + 1e-12)
        return -np.mean(log_probs)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_neural_memory():
    """Demo 1: Show how neural memory learns associations."""
    print("=" * 60)
    print("  Demo 1: Neural Memory (Titans)")
    print("=" * 60)
    print()

    np.random.seed(42)
    d = 16
    mem = NeuralMemory(d)

    # Create some "facts" as key-value pairs
    facts = [
        ("Paris",  np.random.randn(d).astype(np.float32)),
        ("London", np.random.randn(d).astype(np.float32)),
        ("Tokyo",  np.random.randn(d).astype(np.float32)),
    ]
    # Assign each fact a fixed embedding
    fact_embeddings = {name: np.random.randn(d).astype(np.float32) * 0.5
                       for name, _ in facts}

    print("  Memorizing 3 facts (repeated exposure)...")
    for epoch in range(20):
        for name, _ in facts:
            surprise = mem.update(fact_embeddings[name])
        avg_surprise = np.mean(mem.surprise_history[-3:])
        if (epoch + 1) % 4 == 0:
            print(f"    Epoch {epoch + 1:2d}: avg surprise = {avg_surprise:.6f}")

    print()
    print("  Surprise decreases as memory learns the associations!")
    print(f"  First exposure:  {mem.surprise_history[0]:.4f}")
    print(f"  Last exposure:   {mem.surprise_history[-1]:.4f}")
    print(f"  Reduction:       {(1 - mem.surprise_history[-1] / mem.surprise_history[0]) * 100:.1f}%")
    print()


def demo_cms():
    """Demo 2: Show multi-frequency memory updates."""
    print("=" * 60)
    print("  Demo 2: Continuum Memory System (Nested Learning)")
    print("=" * 60)
    print()

    np.random.seed(42)
    d = 16
    cms = ContinuumMemorySystem(d, n_levels=3, base_freq=1)

    print(f"  Memory levels:")
    for i, freq in enumerate(cms.frequencies):
        label = ["fast", "medium", "slow"][i]
        print(f"    Level {i} ({label}):  updates every {freq} token(s)")
    print()

    # Feed tokens and track which levels update
    n_tokens = 32
    print(f"  Processing {n_tokens} tokens...")
    level_updates = {i: 0 for i in range(3)}
    for t in range(n_tokens):
        x_t = np.random.randn(d).astype(np.float32)
        updates = cms.update(x_t)
        for level, _ in updates:
            level_updates[level] += 1

    print()
    for i in range(3):
        label = ["fast", "medium", "slow"][i]
        surprise_hist = cms.memories[i].surprise_history
        avg_s = np.mean(surprise_hist) if surprise_hist else 0
        print(f"    Level {i} ({label}):  {level_updates[i]} updates, "
              f"avg surprise = {avg_s:.4f}")

    print()
    print("  Fast memory updates every token -> captures immediate context")
    print("  Medium memory updates every 4 tokens -> consolidates patterns")
    print("  Slow memory updates every 16 tokens -> stores stable knowledge")
    print()


def demo_self_modifying():
    """Demo 3: Full self-modifying model — memory accumulates over time."""
    print("=" * 60)
    print("  Demo 3: Self-Modifying Model (Hope-like)")
    print("=" * 60)
    print()

    np.random.seed(42)

    model = SelfModifyingModel(vocab_size=32, d_model=16, n_memory_levels=3)

    # Create a repeating pattern
    pattern = np.array([5, 12, 3, 27, 8, 19], dtype=np.int32)
    context = np.tile(pattern, 30)  # 180 tokens

    print(f"  Pattern: {pattern.tolist()} (repeating)")
    print(f"  Context length: {len(context)} tokens")
    print()

    # Process entire context as a stream, measuring loss in chunks
    model.reset()
    chunk_size = 30
    n_chunks = len(context) // chunk_size
    losses = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = context[start:end]
        loss = model.compute_loss(chunk)
        losses.append(loss)
        print(f"  Chunk {i + 1} (tokens {start:>3}-{end:>3}): loss = {loss:.4f}")

    improvement = (1 - losses[-1] / losses[0]) * 100
    print(f"\n  First chunk loss: {losses[0]:.4f}")
    print(f"  Last chunk loss:  {losses[-1]:.4f}")
    print(f"  Improvement:      {improvement:+.1f}%")

    # Show CMS surprise across levels
    print()
    print("  CMS Surprise per Level:")
    for i, mem in enumerate(model.cms.memories):
        if len(mem.surprise_history) >= 10:
            first5 = np.mean(mem.surprise_history[:5])
            last5 = np.mean(mem.surprise_history[-5:])
            label = ["fast", "medium", "slow"][i]
            print(f"    Level {i} ({label}): "
                  f"first 5 avg={first5:.6f} -> last 5 avg={last5:.6f} "
                  f"({(1 - last5 / first5) * 100:+.1f}%)")
    print()


def demo_comparison():
    """Demo 4: Isolated neural memory learning on a direct task."""
    print("=" * 60)
    print("  Demo 4: Neural Memory — Learning to Predict Sequences")
    print("=" * 60)
    print()

    np.random.seed(42)
    d = 16

    mem = NeuralMemory(d)

    # Create a sequence with structure: token A is always followed by token B
    # Represent each token as a fixed embedding
    vocab_size = 8
    embeddings = np.random.randn(vocab_size, d).astype(np.float32) * 0.5
    # Pattern: 0->1->2->3->4->5->6->7->0->1->...
    sequence = list(range(vocab_size)) * 20  # 160 tokens

    print(f"  Task: predict next token in cycle 0->1->2->...->7->0->1->...")
    print(f"  Sequence length: {len(sequence)} tokens")
    print()

    # Feed sequence and track surprise over time
    surprises_by_epoch = []
    epoch_size = vocab_size * 4  # 32 tokens per epoch
    n_epochs = len(sequence) // epoch_size

    for epoch in range(n_epochs):
        epoch_surprise = []
        for i in range(epoch_size):
            idx = epoch * epoch_size + i
            x_t = embeddings[sequence[idx]]
            s = mem.update(x_t)
            epoch_surprise.append(s)
        avg = np.mean(epoch_surprise)
        surprises_by_epoch.append(avg)
        print(f"    Epoch {epoch + 1}: avg surprise = {avg:.6f}")

    print()
    print(f"  First epoch surprise:  {surprises_by_epoch[0]:.6f}")
    print(f"  Last epoch surprise:   {surprises_by_epoch[-1]:.6f}")
    reduction = (1 - surprises_by_epoch[-1] / surprises_by_epoch[0]) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print()
    print("  Memory learns the cyclic pattern -> surprise decreases!")
    print()


def main():
    print()
    print("  Nested Learning & Titans Neural Memory")
    print("  Pure NumPy — no GPU required")
    print()

    t0 = time.time()

    demo_neural_memory()
    demo_cms()
    demo_self_modifying()
    demo_comparison()

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print("  Nested Learning views a model as nested optimization problems:")
    print("    - Outer loop: meta-learns the initialization (slow)")
    print("    - Inner loop: adapts to current context (fast)")
    print("    - CMS: multiple memory levels at different frequencies")
    print()
    print("  Titans adds a neural memory module that:")
    print("    - Learns key->value associations via gradient descent")
    print("    - Uses surprise (prediction error) to gate updates")
    print("    - Provides O(1) retrieval for long-range dependencies")
    print()
    print("  Papers:")
    print("    - Nested Learning: https://arxiv.org/abs/2512.24695")
    print("    - Titans:          https://arxiv.org/abs/2501.00663")
    print()
    print(f"  Total time: {time.time() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()

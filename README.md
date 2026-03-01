# AI Agent Research Summarizer

**GitHub**: https://github.com/dragonfly90/agent_research

**Related**: [JAX Learning Tutorials](https://github.com/dragonfly90/jax_learning) — 13 hands-on JAX/Flax tutorials from basics to distributed training (vmap, pmap, sharding, flash attention, transformers, TPU)

An automated agent that searches for and summarizes the latest AI agent research papers and startups using web search and LLM-powered summarization.

## Features

- **Web Search**: Automatically searches for AI agent research papers and startups
- **LLM Summarization**: Uses Groq's Llama 3.3 70B to generate comprehensive summaries
- **Markdown Reports**: Outputs clean, structured markdown reports
- **Free & Open**: Uses free APIs and open-source models

## Memory Systems for AI Agents

| Layer | Type | Mechanism | Human Analogy |
|-------|------|-----------|---------------|
| Short-Term | Context Window | KV Cache (Sliding Window Attention) | Working Memory |
| Episodic | Architectural | Memory Slots / Recurrent States (e.g., Memformer) | Immediate Recall |
| Long-Term | External | RAG / Vector DBs / Knowledge Graphs | Reference Books |
| Parametric | Internal | Test-Time Training (TTT) / LoRA Adapters | Deep Intuition/Instinct |

## External Memory Systems: Production Frameworks

> 📄 **Full Deep Dive**: [memory_systems_research.md](memory_systems_research.md) — detailed architecture diagrams, benchmarks, and comparison tables for all 5 systems

| System | Architecture | Key Innovation | Open Source |
|--------|-------------|----------------|-------------|
| **[Mem0](https://mem0.ai)** | Extract-Update pipeline + Hybrid DB (Vector+Graph+KV) | Dual-phase memory consolidation; 91% faster, 90% less tokens vs. full context | ✅ ($24M funded, Oct 2025) |
| **[Zep](https://getzep.com)** | Temporal Knowledge Graph (Graphiti engine) | Bi-temporal model captures *when* things happened; +18.5% LongMemEval accuracy | Partially (core Apache 2.0) |
| **[Letta](https://letta.ai)** | LLM-OS / self-editing memory blocks (formerly MemGPT) | The **agent itself** manages memory via tool calls — core/recall/archival hierarchy | ✅ (UC Berkeley + Stanford) |
| **[EverMemOS](https://evermind.ai)** | 4-layer brain-inspired OS (Agentic/Memory/Index/API) | Active memory with fusion+decision mechanisms; 92.3% LoCoMo, 70% token savings | ✅ (Released Dec 2025, backed by OpenAI) |
| **[OpenViking](https://github.com/volcengine/OpenViking)** | Hierarchical Virtual File System (`viking://` URIs) | Tiered context loading L0/L1/L2; recursive retrieval vs. flat RAG | ✅ (ByteDance Volcengine) |

## Test-Time Training (TTT): The "Live" Weight Update

TTT is a cutting-edge technique where an LLM updates its own weights while processing a prompt.

**The Problem**: Traditional Transformers must look back at every token in their context (O(n^2) complexity). With a 1-million-token document, the compute cost is astronomical.

**The TTT Solution**: Instead of storing tokens in a cache, the model treats the context as training data. It performs gradient steps to compress the document into its weights, effectively learning the content without needing to re-read it.

### E2E-TTT (End-to-End Test-Time Training)

> **Paper**: [End-to-End Test-Time Training for Long Context](https://test-time-training.github.io/e2e.pdf)
> **arXiv**: [2512.23675](https://arxiv.org/abs/2512.23675)
> **Authors**: Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rod, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin et al. (Astera Institute, NVIDIA, Stanford, UC Berkeley, UC San Diego)
> **Code**: [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e) (official JAX implementation)

E2E-TTT formulates long-context language modeling as **continual learning** rather than architecture design. Key ideas:

- Uses a standard Transformer with **sliding-window attention** (window size k=8K)
- At test time, performs **mini-batch gradient descent** (batch size b=1K) on the next-token prediction loss to compress context into weights
- Only updates MLP layers in the **last 25% of Transformer blocks** (dual MLP design: one adapts, one stays frozen)
- At training time, uses **meta-learning** to optimize the initialization for test-time learning
- Achieves **constant inference latency** regardless of context length (2.7x faster than full attention at 128K context)
- For 3B models, scales with context length the same way as full attention Transformers

A simplified PyTorch implementation is available in [`ttt_e2e.py`](ttt_e2e.py).

## Continuous and Lifelong Learning

Continuous Learning (CL) focuses on how a model retains knowledge over months or years without Catastrophic Forgetting.

**Key Strategies:**

- **Nested Learning**: Models are designed as a "nest" of optimization problems. Different layers update at different speeds: fast layers adapt to the current conversation, while slow layers consolidate meaningful patterns into permanent memory.
  > **Paper**: [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf) — [arXiv:2512.24695](https://arxiv.org/abs/2512.24695) (Behrouz et al., Google Research, NeurIPS 2025)
  > **Code**: [github.com/obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning), [github.com/WindOfNature/Nested-Learning](https://github.com/WindOfNature/Nested-Learning)
- **Titans**: Associative memory modules that learn to memorize at test time. When the model encounters surprising (high-perplexity) information, it triggers a memory update via gradient descent on the prediction error.
  > **Paper**: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Behrouz, Zhong, Mirrokni — Google Research, 2025)
- **Token-Space Learning**: Some systems (like MemGPT) treat memory as a "file system". The model learns to read/write its own history, performing continuous learning through externalized context management.

A simplified implementation of Nested Learning + Titans is available in [`nested_learning.py`](nested_learning.py).

## Prerequisites

- Python 3.9+
- [Groq API Key](https://console.groq.com) (free)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dragonfly90/agent_research.git
   cd agent_research
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

4. **Run the research agent**:
   ```bash
   python3 main.py
   ```

5. **Run the E2E-TTT demo** (no API key needed, pick any version):
   ```bash
   # Pure NumPy (no dependencies beyond numpy)
   python3 ttt_e2e.py

   # PyTorch version
   pip install torch
   python3 ttt_e2e_torch.py

   # JAX/Flax version
   pip install jax jaxlib flax optax
   python3 ttt_e2e_jax.py

   # Nested Learning & Titans neural memory (pure NumPy)
   python3 nested_learning.py
   ```

## How It Works

1. **Search**: Uses DuckDuckGo to find recent AI agent research and startups
2. **Summarize**: Sends results to Groq's Llama 3.3 70B for intelligent summarization
3. **Report**: Generates a markdown report saved to `research_report.md`

## Configuration

Edit the search queries in `searcher.py` to customize what the agent searches for:

```python
def search_research_papers(query="latest AI agent research papers", max_results=5):
def search_startups(query="new AI agent startups 2024 2025", max_results=5):
```

## Project Structure

| File | Description |
|------|-------------|
| [main.py](https://github.com/dragonfly90/agent_research/blob/main/main.py) | Main application entry point |
| [searcher.py](https://github.com/dragonfly90/agent_research/blob/main/searcher.py) | Web search functionality (DuckDuckGo) |
| [summarizer.py](https://github.com/dragonfly90/agent_research/blob/main/summarizer.py) | LLM summarization (Groq) |
| [forbes_researcher.py](https://github.com/dragonfly90/agent_research/blob/main/forbes_researcher.py) | Forbes AI 50 company researcher |
| [ttt_e2e.py](https://github.com/dragonfly90/agent_research/blob/main/ttt_e2e.py) | E2E-TTT implementation (pure NumPy) |
| [ttt_e2e_torch.py](https://github.com/dragonfly90/agent_research/blob/main/ttt_e2e_torch.py) | E2E-TTT implementation (PyTorch) |
| [ttt_e2e_jax.py](https://github.com/dragonfly90/agent_research/blob/main/ttt_e2e_jax.py) | E2E-TTT implementation (JAX/Flax) |
| [nested_learning.py](https://github.com/dragonfly90/agent_research/blob/main/nested_learning.py) | Nested Learning & Titans neural memory (pure NumPy) |
| [requirements.txt](https://github.com/dragonfly90/agent_research/blob/main/requirements.txt) | Python dependencies |
| [.env.example](https://github.com/dragonfly90/agent_research/blob/main/.env.example) | Environment variable template |
| [research_report.md](https://github.com/dragonfly90/agent_research/blob/main/research_report.md) | Generated AI agent research report |
| [forbes_ai_50_report.md](https://github.com/dragonfly90/agent_research/blob/main/forbes_ai_50_report.md) | Forbes AI 50 companies report |
| [memory_systems_research.md](https://github.com/dragonfly90/agent_research/blob/main/memory_systems_research.md) | Deep dive: Mem0, Zep, Letta, EverMemOS, OpenViking |

## Technologies Used

- **[ddgs](https://github.com/deedy5/ddgs)**: DuckDuckGo search
- **[Groq](https://groq.com)**: Fast LLM inference (Llama 3.3 70B)
- **[PyTorch](https://pytorch.org)**: E2E-TTT implementation (torch version)
- **[JAX](https://github.com/jax-ml/jax)** / **[Flax](https://github.com/google/flax)**: E2E-TTT implementation (JAX version)
- **Python 3**: Core language

## References

- [End-to-End Test-Time Training for Long Context](https://test-time-training.github.io/e2e.pdf) (Tandon et al., 2025) — [arXiv](https://arxiv.org/abs/2512.23675) | [Code](https://github.com/test-time-training/e2e)
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) (Sun et al., 2024)
- [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf) (Behrouz et al., NeurIPS 2025) — [arXiv](https://arxiv.org/abs/2512.24695) | [Code](https://github.com/obekt/HOPE-nested-learning)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Behrouz, Zhong, Mirrokni, 2025)
- [Google Research Blog: Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)

### External Memory Systems
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://github.com/mem0ai/mem0) — $24M funded (Oct 2025)
- [Graphiti: A Temporal Knowledge Graph for AI Agents](https://arxiv.org/abs/2501.13956) (Zep, 2025) — +18.5% LongMemEval accuracy
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) — now [Letta](https://letta.ai)
- [EverMemOS: Brain-Inspired Memory OS for AI Agents](https://evermind.ai) — 92.3% LoCoMo benchmark (EverMind, Dec 2025)
- [OpenViking: Virtual File System Context Database for AI Agents](https://github.com/volcengine/OpenViking) — ByteDance Volcengine, `viking://` URI paradigm



## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Feel free to open issues or submit PRs.

## Troubleshooting

- **No results found**: The search engine might be rate-limiting. Try again later or add delays.
- **Invalid API Key**: Make sure your `.env` file has `GROQ_API_KEY=gsk_...` with your actual key.
- **Model errors**: Check [Groq's documentation](https://console.groq.com/docs) for available models.

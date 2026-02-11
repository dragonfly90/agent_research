# AI Agent Research Summarizer

## memory systems
Layer,Type,Mechanism,Human Analogy
Short-Term,Context Window,KV Cache (Sliding Window Attention),Working Memory
Episodic,Architectural,"Memory Slots / Recurrent States (e.g., Memformer)",Immediate Recall
Long-Term,External,RAG / Vector DBs / Knowledge Graphs,Reference Books
Parametric,Internal,Test-Time Training (TTT) / LoRA Adapters,Deep Intuition/Instinct

## Test-Time Training (TTT): The "Live" Weight Update
TTT is the cutting-edge technique where an LLM updates its own weights while it is processing a prompt.The Problem: Traditional Transformers must look back at every token in their context ($O(n^2)$ complexity). If you have a 1-million-token document, the compute cost is astronomical.The TTT Solution: Instead of "storing" tokens in a cache, the model treats the context as training data. It performs a small number of gradient steps to "compress" the document into a set of Fast Weights (often in the MLP layers).E2E-TTT (End-to-End): Research (e.g., from NVIDIA/Stanford) has introduced architectures where the model's hidden state is a machine learning model itself. As the model reads, it updates its internal representation to predict the next token better, effectively learning the "vibe" or specific facts of a long document without needing to re-read it.

## Continuous and Lifelong Learning
Continuous Learning (CL) focuses on how a model retains knowledge over months or years without Catastrophic Forgetting—the tendency of a model to "overwrite" old knowledge when learning new things.

Key Strategies in 2026:
Nested Learning: Models are designed as a "nest" of optimization problems. Different layers update at different speeds: fast layers adapt to the current conversation, while slow layers consolidate meaningful patterns into permanent memory.

Titans & MIRAS: These frameworks use "associative memory" modules. When the model encounters "surprising" or high-perplexity information, it triggers a memory update, ensuring that unique experiences are prioritized for long-term storage.

Token-Space Learning: Since weight updates can be unstable, some systems (like MemGPT) treat memory as a "file system" (e.g., agents.md). The model learns to read/write its own history, effectively performing continuous learning through externalized, human-readable context management.


An automated agent that searches for and summarizes the latest AI agent research papers and startups using web search and LLM-powered summarization.

## Features

- 🔍 **Web Search**: Automatically searches for AI agent research papers and startups
- 🤖 **LLM Summarization**: Uses Groq's Llama 3.3 70B to generate comprehensive summaries
- 📝 **Markdown Reports**: Outputs clean, structured markdown reports
- 🆓 **Free & Open**: Uses free APIs and open-source models

## Prerequisites

- Python 3.9+
- [Groq API Key](https://console.groq.com) (free)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-agent-researcher.git
   cd ai-agent-researcher
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

4. **Run the agent**:
   ```bash
   python3 main.py
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

```
.
├── main.py              # Main application entry point
├── searcher.py          # Web search functionality
├── summarizer.py        # LLM summarization
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── research_report.md   # Generated report (gitignored)
```

## Technologies Used

- **[ddgs](https://github.com/deedy5/ddgs)**: DuckDuckGo search
- **[Groq](https://groq.com)**: Fast LLM inference (Llama 3.3 70B)
- **Python 3**: Core language

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Feel free to open issues or submit PRs.

## Troubleshooting

- **No results found**: The search engine might be rate-limiting. Try again later or add delays.
- **Invalid API Key**: Make sure your `.env` file has `GROQ_API_KEY=gsk_...` with your actual key.
- **Model errors**: Check [Groq's documentation](https://console.groq.com/docs) for available models.

# AI Agent Research Summarizer

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

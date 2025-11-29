import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def summarize_results(search_results, topic="AI Agent Research"):
    """Summarizes search results using Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found in environment variables."

    client = Groq(api_key=api_key)

    prompt = f"""Please summarize the following search results about {topic}. 
    
    Requirements:
    1. **Startups**: For each startup, provide:
       - Name
       - **Website Link** (if found)
       - **Revenue** (if found)
       - **Funding** (if found)
       - Brief description
    2. **Big Tech**: Create a section for "Big Tech AI Agent Products" (e.g., Google Vertex AI, Azure).
    3. **Trends**: Key findings and trends.
    
    Provide the output in Markdown format.

    Search Results:
    {search_results}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {e}"

if __name__ == "__main__":
    # Test the summarizer (mock data)
    mock_results = [
        {'title': 'Agent AI: Surveying the Horizons of Multimodal Interaction', 'body': 'A comprehensive survey of agent AI...'},
        {'title': 'New Startup: Agentic', 'body': 'Agentic raises $10M to build autonomous agents...'}
    ]
    print(summarize_results(mock_results))

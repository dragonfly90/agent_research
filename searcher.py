from ddgs import DDGS

def search_research_papers(query="latest AI agent research papers", max_results=5):
    """Searches for research papers."""
    print(f"Searching for research papers with query: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                print("No results found.")
            return results
    except Exception as e:
        print(f"Error searching research papers: {e}")
        return []

import time

def search_startups(query="new AI agent startups 2024 2025 funding revenue", max_results=5):
    """Searches for startups with funding and revenue info."""
    print(f"Searching for startups with query: {query}")
    time.sleep(2) # Avoid rate limits
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                print("No results found.")
            return results
    except Exception as e:
        print(f"Error searching startups: {e}")
        return []

def search_big_tech_products(query="big tech AI agent products Google Vertex AI Azure OpenAI AWS Bedrock", max_results=5):
    """Searches for Big Tech AI agent products."""
    print(f"Searching for Big Tech products with query: {query}")
    time.sleep(2)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                print("No results found.")
            return results
    except Exception as e:
        print(f"Error searching big tech products: {e}")
        return []

if __name__ == "__main__":
    # Test the searcher
    print("--- Research Papers ---")
    papers = search_research_papers()
    for p in papers:
        print(p)
    
    print("\n--- Startups ---")
    startups = search_startups()
    for s in startups:
        print(s)

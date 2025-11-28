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

def search_startups(query="new AI agent startups 2024 2025", max_results=5):
    """Searches for startups."""
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

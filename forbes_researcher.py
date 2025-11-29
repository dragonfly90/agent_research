import time
from searcher import search_startups
from summarizer import summarize_results
from dotenv import load_dotenv

# List extracted from Forbes 2025 AI 50 article chunk
FORBES_AI_50_COMPANIES = [
    "Anysphere", 
    "Speak", 
    "OpenEvidence", 
    "OpenAI", 
    "Anthropic", 
    "xAI", 
    "Thinking Machine Labs", 
    "World Labs", 
    "Writer", 
    "Crusoe", 
    "Lambda", 
    "Together AI"
]

def research_company(company_name):
    """Researches a specific company."""
    query = f"{company_name} AI startup revenue funding website"
    print(f"Researching {company_name}...")
    results = search_startups(query, max_results=3)
    return results

def main():
    load_dotenv()
    print(f"Starting research on {len(FORBES_AI_50_COMPANIES)} Forbes AI 50 companies...")
    
    all_company_data = []
    
    for company in FORBES_AI_50_COMPANIES:
        data = research_company(company)
        if data:
            all_company_data.append(f"## {company}\n{data}\n")
        time.sleep(1) # Be nice to the search engine

    if not all_company_data:
        print("No data found.")
        return

    print("\nSummarizing all findings...")
    full_text = "\n".join(all_company_data)
    
    # We might need to chunk this if it's too long, but for 12 companies it should be fine for 70B model
    summary = summarize_results(full_text, topic="Forbes AI 50 2025 Startups")
    
    with open("forbes_ai_50_report.md", "w") as f:
        f.write(summary)
    
    print("Done! Report saved to 'forbes_ai_50_report.md'.")

if __name__ == "__main__":
    main()

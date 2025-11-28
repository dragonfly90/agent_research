import os
from searcher import search_research_papers, search_startups
from summarizer import summarize_results
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("Starting AI Agent Research...")
    
    # 1. Search for Research Papers
    print("\n[1/3] Searching for Research Papers...")
    papers = search_research_papers()
    
    # 2. Search for Startups
    print("\n[2/3] Searching for Startups...")
    startups = search_startups()
    
    if not papers and not startups:
        print("No information found. Exiting.")
        return

    # 3. Summarize Findings
    print("\n[3/3] Summarizing Findings...")
    all_results = f"Research Papers:\n{papers}\n\nStartups:\n{startups}"
    summary = summarize_results(all_results)
    
    # 4. Save Report
    with open("research_report.md", "w") as f:
        f.write(summary)
    
    print("\nDone! Report saved to 'research_report.md'.")
    print("-" * 20)
    print(summary)

if __name__ == "__main__":
    main()

import os
from searcher import search_research_papers, search_startups, search_big_tech_products
from summarizer import summarize_results
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("Starting AI Agent Research...")
    
    # 1. Search for Research Papers
    print("\n[1/4] Searching for Research Papers...")
    papers = search_research_papers()
    
    # 2. Search for Startups
    print("\n[2/4] Searching for Startups...")
    startups = search_startups()

    # 3. Search for Big Tech
    print("\n[3/4] Searching for Big Tech Products...")
    big_tech = search_big_tech_products()
    
    if not papers and not startups and not big_tech:
        print("No information found. Exiting.")
        return

    # 4. Summarize Findings
    print("\n[4/4] Summarizing Findings...")
    all_results = f"Research Papers:\n{papers}\n\nStartups:\n{startups}\n\nBig Tech Products:\n{big_tech}"
    summary = summarize_results(all_results)
    
    # 4. Save Report
    with open("research_report.md", "w") as f:
        f.write(summary)
    
    print("\nDone! Report saved to 'research_report.md'.")
    print("-" * 20)
    print(summary)

if __name__ == "__main__":
    main()

import os
import time
# Importing the functional logic we built in the previous steps
from paper_engine import search_arxiv
from download_engine import download_paper_pdf

def run_search_and_download_pipeline(topic_query: str, num_papers: int = 2):
    """Executes the complete sequence: searches arXiv and downloads matching papers."""
    print(f"🚀 Starting Research Pipeline for topic: '{topic_query}'")
    
    # Step 1: Execute search using our P1 Engine
    found_papers = search_arxiv(topic_query, max_results=num_papers)
    
    if not found_papers:
        print("❌ No papers found matching the query criteria.")
        return
        
    print(f"📋 Found {len(found_papers)} papers. Beginning download sequence...")
    
    # Step 2: Loop through discovered items and trigger P3 Downloader
    for idx, paper in enumerate(found_papers, start=1):
        print(f"\n[Processing {idx}/{len(found_papers)}] {paper['id']}")
        print(f"📖 Title: {paper['title']}")
        
        # Connect search data directly to download engine
        saved_file_path = download_paper_pdf(paper['id'])
        
        if saved_file_path:
            print(f"📁 Local File Target: {saved_file_path}")
        else:
            print(f"⚠️ Failed to securely cache PDF for ID: {paper['id']}")
            
    print("\n🏁 Pipeline execution complete!")

# --- TEST EXECUTION GATE ---
if __name__ == "__main__":
    # Test it live with a modern research query
    run_search_and_download_pipeline("Deep learning for medical image segmentation", num_papers=5)

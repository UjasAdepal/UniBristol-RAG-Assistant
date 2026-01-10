import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- CONFIGURATION ---
OUTPUT_FILE = "general_knowledge_archive.json"

# The URLs we want to "Time Travel" to
SEEDS = [
    {"url": "https://www.bristol.ac.uk/accommodation/undergraduate/", "category": "Accommodation"},
    {"url": "https://www.bristol.ac.uk/students/new/", "category": "New Students"},
    {"url": "https://www.bristol.ac.uk/students/support/", "category": "Student Support"},
    {"url": "https://www.bristol.ac.uk/students/support/finances/", "category": "Money & Fees"},
    {"url": "https://www.bristol.ac.uk/students/visa/", "category": "Visas & International"},
]

def get_wayback_url(original_url):
    """Asks Archive.org for the latest snapshot of the page."""
    api_url = f"http://archive.org/wayback/available?url={original_url}"
    try:
        r = requests.get(api_url, timeout=10)
        data = r.json()
        if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
            return data["archived_snapshots"]["closest"]["url"]
    except:
        pass
    return None

def scrape_archive_page(target_url, category, depth=1):
    # 1. Get the Snapshot URL
    print(f"⏳ Finding archive for: {target_url}...", end=" ")
    archive_url = get_wayback_url(target_url)
    
    if not archive_url:
        print("❌ Not found in archive")
        return None, []

    # 2. Scrape the Snapshot
    try:
        # We use standard requests because Archive.org is friendly to bots
        response = requests.get(archive_url, timeout=15)
        
        soup = BeautifulSoup(response.content, "html.parser")

        # 3. Clean up the "Wayback Machine" Toolbar (The blue banner)
        for tag in soup.find_all(id=lambda x: x and "wm-ipp" in x):
            tag.decompose()
        for tag in soup(['nav', 'footer', 'script', 'style', 'header', 'aside']):
            tag.decompose()

        # 4. Extract Text
        main = soup.find('main') or soup.find('div', id='content') or soup.find('body')
        text = main.get_text(separator="\n\n", strip=True)
        title = soup.title.string.strip() if soup.title else category

        # 5. Spider: Find Internal Links (rewritten to point to Archive.org)
        # We only want links that point back to bristol.ac.uk
        links = []
        if depth == 1: # Only look for children on the first pass
            for a in main.find_all('a', href=True):
                href = a['href']
                # Archive.org rewrites links like: /web/2023.../https://bristol.ac.uk/foo
                # We want to extract the REAL link to find ITS archive later
                if "bristol.ac.uk" in href and "http" in href:
                    # Clean the URL to get the original Bristol link
                    real_link = href.split("http")[-1]
                    real_link = "http" + real_link
                    links.append(real_link)

        print("✅ Success")
        return {
            "title": title,
            "url": target_url, # Store the REAL url, not the archive one
            "category": category,
            "content": text
        }, links

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None, []

def main():
    scraped_data = []
    
    # Queue: (url, category, depth)
    queue = []
    for seed in SEEDS:
        queue.append((seed['url'], seed['category'], 1))

    # We keep track of visited URLs so we don't loop
    visited = set()
    
    # Simple BFS Crawl
    max_pages = 50
    
    while queue and len(scraped_data) < max_pages:
        url, category, depth = queue.pop(0)
        
        if url in visited: continue
        visited.add(url)

        data, children = scrape_archive_page(url, category, depth)
        
        if data and len(data['content']) > 200:
            scraped_data.append(data)
            
            # Add children to queue
            if depth < 2:
                # Limit to 5 children per page to keep it broad
                for child_url in children[:5]:
                    if child_url not in visited:
                        queue.append((child_url, category, depth + 1))
        
        # Archive.org can be slow, be nice
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4)
        
    print(f"🎉 Done! Saved {len(scraped_data)} archived pages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
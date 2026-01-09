import json
import time
import random
import os
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
INPUT_FILE = "bristol_target_urls.json"  # The file you uploaded
OUTPUT_FILE = "general_knowledge_massive.json"
MAX_PAGES_PER_CATEGORY = 20 # Limit to avoid scraping 5,000 pages

# --- YOUR LOGIC (Resurrected) ---
def infer_topic(url):
    """
    Your original logic to categorize pages based on URL.
    """
    path = urlparse(url).path.lower()
    if "accommodation" in path or "residences" in path:
        return "Accommodation"
    elif "fees" in path or "funding" in path or "finance" in path or "cost" in path:
        return "Fees and Funding"
    elif "support" in path or "counselling" in path or "wellbeing" in path or "health" in path:
        return "Student Support"
    elif "visa" in path or "international" in path:
        return "Visas & International"
    elif "rules" in path or "regulations" in path or "handbook" in path:
        return "Rules & Regulations"
    elif "sport" in path or "gym" in path:
        return "Sports & Lifestyle"
    elif "students/new" in path or "welcome" in path:
        return "New Students"
    else:
        return "Other" # We will skip these for now to save time

def scrape_page_stealth(page, url):
    """
    Uses Playwright Stealth to bypass the firewall.
    """
    print(f"🕷️ Visiting: {url}...", end=" ")
    try:
        # 1. Navigate
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        
        # 2. Check for Blocks
        if "blocked" in page.title().lower() or "security" in page.title().lower():
            print("🚫 BLOCKED! Skipping...")
            return None

        # 3. Human Behavior (Scroll & Jiggle)
        page.mouse.move(random.randint(100, 500), random.randint(100, 500))
        page.mouse.wheel(0, 1000)
        time.sleep(random.uniform(2, 4))

        # 4. Expand Accordions (Crucial for FAQs)
        try:
            page.evaluate("""() => {
                document.querySelectorAll('button[aria-expanded="false"]').forEach(b => b.click());
                document.querySelectorAll('details').forEach(d => d.open = true);
            }""")
            time.sleep(1)
        except: pass

        # 5. Extract
        soup = BeautifulSoup(page.content(), "html.parser")
        
        # Cleanup
        for tag in soup(['nav', 'footer', 'script', 'style', 'header', 'form']):
            tag.decompose()
            
        main = soup.find('main') or soup.find('div', id='content') or soup.find('body')
        text = main.get_text(separator="\n\n", strip=True)
        title = soup.title.string.strip() if soup.title else "Unknown"
        
        print(f"✅ Success ({len(text)} chars)")
        return {
            "title": title,
            "url": url,
            "content": text
        }

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

def main():
    # 1. Load the Big List
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File {INPUT_FILE} not found. Please create it with your URL list.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_urls = json.load(f)
    
    # 2. Filter & Prioritize
    print(f"🔍 Analyzing {len(raw_urls)} URLs...")
    targets = {}
    
    for url in raw_urls:
        topic = infer_topic(url)
        if topic != "Other":
            if topic not in targets:
                targets[topic] = []
            if len(targets[topic]) < MAX_PAGES_PER_CATEGORY:
                targets[topic].append(url)
    
    total_targets = sum(len(v) for v in targets.values())
    print(f"🎯 Selected {total_targets} high-value pages to scrape.")
    for topic, urls in targets.items():
        print(f"   - {topic}: {len(urls)} pages")

    # 3. Launch the Stealth Scraper
    scraped_data = []
    
    with sync_playwright() as p:
        # Headless=False is vital for Bristol's firewall
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context(
             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        stealth_sync(page)

        for topic, urls in targets.items():
            print(f"\n--- Processing Category: {topic} ---")
            for url in urls:
                data = scrape_page_stealth(page, url)
                if data:
                    data['category'] = topic # Add your category tag
                    scraped_data.append(data)
                
                # Save periodically
                if len(scraped_data) % 5 == 0:
                     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(scraped_data, f, indent=4)
                
                # Sleep to be polite
                time.sleep(random.uniform(2, 5))

        browser.close()

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4)
    print(f"🎉 Done! Saved {len(scraped_data)} pages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
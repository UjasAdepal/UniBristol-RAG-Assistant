import json
import os
import time
import random
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
INPUT_FILES = [
    "enriched_undergraduate_courses.json",
    "enriched_postgraduate_courses2.json"
]
OUTPUT_FILE = "final_course_data_full.json"

def get_existing_urls(files):
    """Extracts URLs from your JSON files."""
    urls = []
    print("🔍 Reading data...")
    for file_path in files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry in data:
                        if "url" in entry and entry["url"]:
                            urls.append({
                                "url": entry["url"], 
                                "title": entry.get("title", "Unknown"),
                                "programme_type": entry.get("programme_type", "Course")
                            })
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    # Remove duplicates
    unique_urls = {u['url']: u for u in urls}.values()
    print(f"✅ Found {len(unique_urls)} unique courses.")
    return list(unique_urls)

def scrape_course(page, url):
    try:
        # 1. Go to page
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        
        # 2. Check blocked status
        if "blocked" in page.title().lower() or "unusual activity" in page.content():
            print("🚫 BLOCKED! Pausing for 60 seconds...")
            time.sleep(60)
            return None

        # 3. HUMAN BEHAVIOR (Essential for strict firewalls)
        # Random mouse movements
        page.mouse.move(random.randint(100, 500), random.randint(100, 500))
        # Scroll to bottom to trigger lazy loading
        page.mouse.wheel(0, 2000) 
        time.sleep(random.uniform(2, 4)) 

        # 4. EXPAND HIDDEN TEXT
        try:
            # Click "Show more" buttons if they exist
            page.evaluate("""() => {
                document.querySelectorAll('button[aria-expanded="false"]').forEach(b => b.click());
                document.querySelectorAll('.accordion__button').forEach(b => b.click());
            }""")
            time.sleep(1) 
        except:
            pass

        # 5. EXTRACTION
        content_html = page.content()
        soup = BeautifulSoup(content_html, "html.parser")
        
        # Priority 1: <main> tag (Best)
        target_section = soup.find("main")
        
        # Priority 2: #content div
        if not target_section:
            target_section = soup.find("div", {"id": "content"})

        if target_section:
            full_text = target_section.get_text(separator="\n\n", strip=True)
            
            # Clean up excessive newlines
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            # Final check for block message in text
            if "Cyber Security systems" in clean_text:
                return None
                
            return clean_text
        else:
            return None

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

def main():
    course_list = get_existing_urls(INPUT_FILES)
    if not course_list:
        return

    scraped_data = []
    
    # Check if we have a partial save file to resume from?
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                scraped_data = json.load(f)
            print(f"♻️  Resuming! Found {len(scraped_data)} already scraped.")
            # Filter out URLs we already have
            scraped_urls = set(item['url'] for item in scraped_data)
            course_list = [c for c in course_list if c['url'] not in scraped_urls]
            print(f"📉 Remaining courses to scrape: {len(course_list)}")
        except:
            print("⚠️ Could not read existing file, starting fresh.")

    with sync_playwright() as p:
        print("🕵️  Launching Browser...")
        # Keep headless=False to look human
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        stealth_sync(page)

        print(f"🕷️ Starting scrape...")
        
        for i, course in enumerate(course_list):
            print(f"[{i+1}/{len(course_list)}] {course['title']}...", end=" ")
            
            # Random delay between requests (3-6 seconds)
            time.sleep(random.uniform(3, 6))
            
            text = scrape_course(page, course["url"])
            
            if text and len(text) > 200:
                print("✅ Success")
                scraped_data.append({
                    "title": course["title"],
                    "url": course["url"],
                    "programme_type": course["programme_type"],
                    "full_content": text 
                })
            else:
                print("❌ Failed/Blocked")

            # Save every 20 courses
            if (i + 1) % 20 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(scraped_data, f, indent=4)
                print("💾 Saved progress.")

    # Final Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4)
    print(f"🎉 Done! All data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    
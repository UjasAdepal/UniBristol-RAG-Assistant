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
OUTPUT_FILE = "stealth_course_data.json"

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
    print(f"✅ Found {len(urls)} courses.")
    return urls



def scrape_course(page, url):
    try:
        # 1. Go to page
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        
        # 2. Check blocked status
        if "blocked" in page.title().lower() or "unusual activity" in page.content():
            print("🚫 BLOCKED! Pausing...")
            time.sleep(30)
            return None

        # 3. HUMAN BEHAVIOR
        page.mouse.move(random.randint(100, 500), random.randint(100, 500))
        page.mouse.wheel(0, 1000) # Scroll down to trigger lazy-loading
        time.sleep(random.uniform(2, 4)) # Wait for bottom content to load

        # 4. EXPAND HIDDEN TEXT (The "Accordion" Fix)
        # Some UoB pages hide structure behind "Show more" or accordion buttons.
        try:
            # Try to click buttons that look like expanders
            page.evaluate("""() => {
                document.querySelectorAll('button[aria-expanded="false"]').forEach(b => b.click());
                document.querySelectorAll('.accordion__button').forEach(b => b.click());
            }""")
            time.sleep(1) # Wait for animation
        except:
            pass

        # 5. SURGICAL EXTRACTION: Target the WHOLE page body
        content_html = page.content()
        soup = BeautifulSoup(content_html, "html.parser")
        
        # Target <main> which usually holds everything (Summary + Structure + Fees)
        target_section = soup.find("main")
        
        # Fallback to #content if <main> is missing
        if not target_section:
            target_section = soup.find("div", {"id": "content"})

        if target_section:
            # Get text with clear separation
            full_text = target_section.get_text(separator="\n\n", strip=True)
            
            # Clean up: Remove huge gaps but keep paragraph structure
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            # Final check for block message
            if "Cyber Security systems" in clean_text:
                return None
                
            return clean_text
        else:
            return None

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None




    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

def main():
    course_list = get_existing_urls(INPUT_FILES)
    if not course_list:
        return

    # TEST WITH FIRST 5 ONLY
    print("⚠️ TEST MODE: Scraping first 5 courses only.")
    course_list = course_list[:5] 

    scraped_data = []
    
    with sync_playwright() as p:
        print("🕵️  Launching Stealth Browser...")
        # Headless=False is CRITICAL for bypassing blocks
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        # ACTIVATE STEALTH MODE
        stealth_sync(page)

        print(f"🕷️ Starting scrape...")
        
        for i, course in enumerate(course_list):
            print(f"[{i+1}] {course['title']}...", end=" ")
            
            # Add random start delay
            time.sleep(random.uniform(2, 5))
            
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=4)
    print(f"🎉 Done! Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
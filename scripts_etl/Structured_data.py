import json
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 🎛️ CONTROL PANEL
# ==========================================
TEST_MODE = False  # Set to True for a cheap test (5 courses). Set to False for the full run.
INPUT_FILE = "final_course_data_full.json"  # The file created by your full scraper
OUTPUT_FILE = "structured_course_data.json" # The final clean file for the vector DB
# ==========================================

# Use GPT-4o-mini (Cheap & Fast)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

# The "Brain" Prompt
prompt = PromptTemplate(
    template="""
    You are an expert data extractor for the University of Bristol.
    I will give you the raw text from a course page.
    Your job is to extract specific details into a clean JSON format.

    Raw Text:
    {raw_text}

    Instructions:
    1. "summary": Write a concise 3-4 sentence summary of what the course is about.
    2. "fees": Extract the specific tuition fees (e.g., "Home: £9,535, Overseas: £26,500"). If not found, write "Check website".
    3. "requirements": Extract the key entry requirements (A-levels, IB, etc.). Keep it readable.
    4. "structure": Summarize the course structure (e.g., "Year 1 focus on X, Year 2 options include Y").
    5. "careers": List specific job roles or companies mentioned.
    6. "contact": Extract email or phone number if available.
    7. Return ONLY the JSON object.

    {format_instructions}
    """,
    input_variables=["raw_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        # Fallback for testing if you haven't finished the full scrape yet
        if os.path.exists("stealth_course_data.json"):
            print("⚠️ 'final_course_data_full.json' not found. Using 'stealth_course_data.json' for testing.")
            file_to_read = "stealth_course_data.json"
        else:
            print(f"❌ Input file {INPUT_FILE} not found.")
            return
    else:
        file_to_read = INPUT_FILE

    with open(file_to_read, "r", encoding="utf-8") as f:
        raw_courses = json.load(f)

    # 2. Apply Test Limits
    if TEST_MODE:
        print(f"🧪 TEST MODE ACTIVE: Processing only first 5 courses out of {len(raw_courses)}.")
        courses_to_process = raw_courses[:5]
        current_output_file = "test_structured_data.json"
    else:
        print(f"🚀 PRODUCTION MODE: Processing ALL {len(raw_courses)} courses.")
        courses_to_process = raw_courses
        current_output_file = OUTPUT_FILE

    structured_data = []

    # Resume capability (only for production mode)
    if not TEST_MODE and os.path.exists(current_output_file):
        with open(current_output_file, "r", encoding="utf-8") as f:
            structured_data = json.load(f)
            print(f"♻️  Resuming... {len(structured_data)} courses already done.")
    
    processed_urls = {c['url'] for c in structured_data}

    print("------------------------------------------------")
    
    for i, course in enumerate(courses_to_process):
        if course['url'] in processed_urls:
            continue

        print(f"[{i+1}/{len(courses_to_process)}] Processing: {course['title']}...", end=" ", flush=True)

        try:
            # Run the AI extraction
            # We truncate input to 15,000 chars to prevent token overflow (saves money too)
            result = chain.invoke({"raw_text": course['full_content'][:15000]})
            
            # Combine
            structured_entry = {
                "title": course['title'],
                "url": course['url'],
                "type": course['programme_type'],
                **result # Unpack AI results
            }
            
            structured_data.append(structured_entry)
            print("✅ Done")

        except Exception as e:
            print(f"❌ Error: {e}")

        # Save frequently
        if (i + 1) % 5 == 0:
            with open(current_output_file, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=4)

    # Final Save
    with open(current_output_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4)
    
    print("------------------------------------------------")
    print(f"🎉 Finished! Saved to: {current_output_file}")
    if TEST_MODE:
        print("💡 Review the file. If it looks good, set TEST_MODE = False and run again.")

if __name__ == "__main__":
    main()
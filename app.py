import os
import json
import re
import sys
import time
from datetime import datetime

from services.ocr_service import extract_text
from services.llm_parser import llm

UPLOAD_FOLDER = "uploads"
STORAGE_DIR = "storage"
COMPETITIONS_FILE = os.path.join(STORAGE_DIR, "competitions.json")
FAILED_OCR_FILE = os.path.join(STORAGE_DIR, "failed_ocr_text.txt")

# ---------------------------
# Helper: clean AI output and extract JSON
# ---------------------------
def extract_json_from_llm_output(text):
    """Extract JSON array or object from markdown-fenced or plain text."""
    text = text.strip()

    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    # Try to find the first '[' or '{' and the matching closing bracket/brace
    start = text.find('[')
    if start != -1:
        bracket_count = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '[':
                bracket_count += 1
            elif ch == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = text[start:i+1]
                    break
        else:
            json_str = None
    else:
        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i, ch in enumerate(text[start:], start):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start:i+1]
                        break
            else:
                json_str = None
        else:
            json_str = None

    if not json_str:
        raise ValueError("No JSON array or object found in LLM output")

    parsed = json.loads(json_str)

    # If it's an object with a "competitions" key, extract that list
    if isinstance(parsed, dict) and "competitions" in parsed:
        parsed = parsed["competitions"]

    if not isinstance(parsed, list):
        parsed = [parsed]

    return parsed

# ---------------------------
# Helper: save a single competition to JSON file
# ---------------------------
def save_competition(competition_data):
    """Append one competition dictionary to competitions.json"""
    os.makedirs(STORAGE_DIR, exist_ok=True)

    if os.path.exists(COMPETITIONS_FILE):
        with open(COMPETITIONS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    competition_data["saved_at"] = datetime.now().isoformat()
    data.append(competition_data)

    with open(COMPETITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved competition: {competition_data.get('competition', 'Unknown')}")

# ---------------------------
# Helper: call LLM with retry + exponential backoff
# ---------------------------
def invoke_llm_with_retry(llm, prompt, max_retries=3):
    """
    Calls the LLM with automatic retry on rate-limit (429) errors.
    Waits progressively longer between attempts.
    """
    for attempt in range(max_retries):
        try:
            print(f"🤖 Calling LLM... (attempt {attempt + 1}/{max_retries})")
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 10s, 20s, 40s...
                wait_time = (2 ** attempt) * 10
                print(f"⏳ Rate limited (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Either not a rate limit, or we're out of retries
                raise

    # Should never reach here, but just in case
    return llm.invoke(prompt)

# ==================================================
# ================= MAIN EXECUTION =================
# ==================================================

all_text = ""

# Read all images from uploads folder
for file_name in os.listdir(UPLOAD_FOLDER):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(UPLOAD_FOLDER, file_name)
        print(f"\nProcessing: {file_name}")
        extracted_text = extract_text(image_path)
        all_text += "\n" + extracted_text

print("\n===== OCR TEXT (first 500 chars) =====\n")
print(all_text[:500] + ("..." if len(all_text) > 500 else ""))

# LLM parsing
prompt = f"""
Extract all case competitions from the OCR text below.

Return ONLY a valid JSON array. Do NOT include markdown formatting, code fences, or any explanatory text.
The response must start with '[' and end with ']'.

Each object must have these fields:
- competition (string)
- registration_deadline (string in YYYY-MM-DD format, use "" if missing)
- event_date (string in YYYY-MM-DD format, use "" if missing)
- website (string, use "" if missing)
- event_type (string, e.g., "Case Competition")

OCR Text:
{all_text}
"""

# Try to call LLM with retry logic
try:
    response = invoke_llm_with_retry(llm, prompt, max_retries=3)
    ai_output = response.content.strip()

    print("\n===== AI RAW OUTPUT =====\n")
    print(ai_output)

except Exception as e:
    print(f"\n❌ LLM call failed after retries: {e}")
    print("💾 Saving extracted OCR text so you don't lose it...")

    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(FAILED_OCR_FILE, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"📝 OCR text saved to: {FAILED_OCR_FILE}")
    print("🔧 Fix your API key / quota and re-run. The script will read from uploads again.")
    sys.exit(1)

# Parse JSON and save each competition
try:
    competitions = extract_json_from_llm_output(ai_output)

    if len(competitions) == 0:
        print("⚠️ No competitions found in the AI output.")
    else:
        for comp in competitions:
            # Ensure required fields exist (fill missing with empty string)
            required = ["competition", "registration_deadline", "event_date", "website", "event_type"]
            for field in required:
                if field not in comp:
                    comp[field] = ""
            save_competition(comp)

        print(f"\n✅ Successfully saved {len(competitions)} competition(s) to {COMPETITIONS_FILE}")

except json.JSONDecodeError as e:
    print(f"❌ Failed to parse extracted JSON: {e}")
    print("Raw AI output was:", ai_output)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
import os
import json
import re
import sys
import time
from datetime import datetime

from services.ocr_service import extract_text
from services.llm_parser import llm

# ---------- NEW: Import search engine ----------
from services.search_engine import CompetitionSearchEngine

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

# ---------------------------
# NEW: Merge search results into competition dict
# ---------------------------
def enrich_competition_with_search(comp_dict, search_engine):
    """
    Call the search engine for the competition name and merge returned fields.
    Only overwrites fields that are missing/empty or where confidence > 0.7.
    """
    name = comp_dict.get("competition", "")
    if not name:
        print("⚠️ Skipping search: competition name missing")
        return comp_dict

    print(f"🔍 Enriching '{name}' via web search...")
    try:
        search_data = search_engine.search(name)
        # search_data is a dict with keys:
        # competition, website, registration_deadline, competition_start_date,
        # competition_end_date, organizer, confidence_score

        # Merge strategy: prefer non-empty search result values,
        # but keep original if it exists and search confidence is low
        if search_data.get("confidence_score", 0) >= 0.5:
            # Website
            if not comp_dict.get("website") and search_data.get("website"):
                comp_dict["website"] = search_data["website"]
            # Registration deadline
            if not comp_dict.get("registration_deadline") and search_data.get("registration_deadline"):
                comp_dict["registration_deadline"] = search_data["registration_deadline"]
            # Event date: search returns competition_start_date; we map to event_date if empty
            if not comp_dict.get("event_date") and search_data.get("competition_start_date"):
                comp_dict["event_date"] = search_data["competition_start_date"]
            # Add new fields that were not originally in OCR schema
            comp_dict["organizer"] = search_data.get("organizer")  # may be None
            comp_dict["end_date"] = search_data.get("competition_end_date")
            comp_dict["search_confidence"] = search_data.get("confidence_score")
        else:
            print(f"   Low confidence ({search_data.get('confidence_score')}) – keeping original data only")
    except Exception as e:
        print(f"   ❌ Search failed for '{name}': {e}")

    return comp_dict

# ==================================================
# ================= MAIN EXECUTION =================
# ==================================================

# ---------- NEW: Initialize the search engine ----------
# It reads GOOGLE_API_KEY from .env (make sure .env is in the expected location)
try:
    search_engine = CompetitionSearchEngine()
    print("✅ CompetitionSearchEngine initialized")
except Exception as e:
    print(f"⚠️ Failed to initialize search engine: {e}")
    print("   Web enrichment will be skipped.")
    search_engine = None

# ---------- OCR extraction ----------
all_text = ""

for file_name in os.listdir(UPLOAD_FOLDER):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(UPLOAD_FOLDER, file_name)
        print(f"\nProcessing: {file_name}")
        extracted_text = extract_text(image_path)
        all_text += "\n" + extracted_text

print("\n===== OCR TEXT (first 500 chars) =====\n")
print(all_text[:500] + ("..." if len(all_text) > 500 else ""))

# ---------- LLM parsing ----------
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

# ---------- Parse and enrich ----------
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

            # NEW: Enrich with web search if engine is available
            if search_engine:
                comp = enrich_competition_with_search(comp, search_engine)

            save_competition(comp)

        print(f"\n✅ Successfully saved {len(competitions)} competition(s) to {COMPETITIONS_FILE}")

except json.JSONDecodeError as e:
    print(f"❌ Failed to parse extracted JSON: {e}")
    print("Raw AI output was:", ai_output)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
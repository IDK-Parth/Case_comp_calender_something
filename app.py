import os
import json
import re
from datetime import datetime

from services.ocr_service import extract_text
from services.llm_parser import llm
from calendar_service import get_calendar_service, create_event

UPLOAD_FOLDER = "uploads"
STORAGE_DIR = "storage"
COMPETITIONS_FILE = os.path.join(STORAGE_DIR, "competitions.json")

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
                    if bracelet_count == 0:
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
# Sync all competitions to Google Calendar
# ---------------------------
def sync_competitions_to_calendar():
    """Read competitions.json and create calendar events (reg deadline + event day)."""
    if not os.path.exists(COMPETITIONS_FILE):
        print("⚠️ No competitions.json found. Nothing to sync to calendar.")
        return

    with open(COMPETITIONS_FILE, "r", encoding="utf-8") as f:
        competitions = json.load(f)

    if not competitions:
        print("⚠️ competitions.json is empty.")
        return

    print("\n🔐 Authenticating with Google Calendar...")
    service = get_calendar_service()

    for comp in competitions:
        name = comp.get("competition", "Unknown")
        reg_deadline = comp.get("registration_deadline")
        event_date = comp.get("event_date")
        website = comp.get("website", "")

        # Registration deadline event
        if reg_deadline and reg_deadline.strip():
            try:
                create_event(
                    service,
                    title=f"{name} - Registration Deadline",
                    date=reg_deadline,
                    description=website
                )
                print(f"✅ Calendar: Added registration deadline for {name}")
            except Exception as e:
                print(f"❌ Failed to add reg deadline for {name}: {e}")
        else:
            print(f"⚠️ No registration_deadline for {name}, skipping deadline event")

        # Competition day event
        if event_date and event_date.strip():
            try:
                create_event(
                    service,
                    title=f"{name} - Competition Day",
                    date=event_date,
                    description=website
                )
                print(f"✅ Calendar: Added competition day for {name}")
            except Exception as e:
                print(f"❌ Failed to add competition day for {name}: {e}")
        else:
            print(f"⚠️ No event_date for {name}, skipping competition day event")

    print("\n🎉 Calendar sync complete!")

# ---------------------------
# Check if LLM processing should be skipped (i.e., we already have data)
# ---------------------------
def should_skip_llm():
    """Return True if competitions.json already exists."""
    return os.path.exists(COMPETITIONS_FILE)

# ==================================================
# ================= MAIN EXECUTION =================
# ==================================================

# ✅ IMPORTANT: Check FIRST before doing any OCR or LLM work
if should_skip_llm():
    print(f"\n⚠️ Found existing {COMPETITIONS_FILE} – skipping OCR and LLM processing.")
    print("Will sync existing competitions to Google Calendar instead.\n")
    sync_competitions_to_calendar()
    import sys
    sys.exit(0)

# --------------------------------------------------
# If we reach here, competitions.json does NOT exist.
# Proceed with full extraction (OCR + LLM + save).
# --------------------------------------------------

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
Extract all case competitions from this text.

Return ONLY valid JSON array.

Each object must have these fields:
- competition (string)
- registration_deadline (string in YYYY-MM-DD format, if missing use empty string)
- event_date (string in YYYY-MM-DD format, if missing use empty string)
- website (string, can be empty)
- event_type (string, e.g., "Case Competition")

OCR Text:
{all_text}
"""

response = llm.invoke(prompt)
ai_output = response.content.strip()

print("\n===== AI RAW OUTPUT =====\n")
print(ai_output)

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

        # After saving, push all competitions to Google Calendar
        sync_competitions_to_calendar()

except json.JSONDecodeError as e:
    print(f"❌ Failed to parse extracted JSON: {e}")
    print("Raw AI output was:", ai_output)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
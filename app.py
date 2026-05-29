import os
import json
import re
from datetime import datetime

from services.ocr_service import extract_text
from services.llm_parser import llm

UPLOAD_FOLDER = "uploads"
STORAGE_DIR = "storage"
COMPETITIONS_FILE = os.path.join(STORAGE_DIR, "competitions.json")

# ---------------------------
# Helper: clean AI output and extract JSON
# ---------------------------
def extract_json_from_llm_output(text):
    """Extract JSON array or object from markdown-fenced or plain text."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # 1. Remove markdown code fences (```json ... ``` or ``` ... ```)
    # Pattern: ```(?:json)?\s*([\s\S]*?)\s*```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    # 2. Try to find the first '[' or '{' and the matching closing bracket/brace
    # First look for array [...]
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
        # If no array, try object {...}
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

    # Parse the extracted JSON string
    parsed = json.loads(json_str)

    # If it's an object with a "competitions" key, extract that list
    if isinstance(parsed, dict) and "competitions" in parsed:
        parsed = parsed["competitions"]

    # Ensure it's a list
    if not isinstance(parsed, list):
        parsed = [parsed]   # single object -> wrap in list

    return parsed


# ---------------------------
# Helper: save a single competition
# ---------------------------
def save_competition(competition_data):
    """Append one competition dictionary to competitions.json"""
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Load existing data
    if os.path.exists(COMPETITIONS_FILE):
        with open(COMPETITIONS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Add timestamp and append
    competition_data["saved_at"] = datetime.now().isoformat()
    data.append(competition_data)

    # Write back
    with open(COMPETITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved competition: {competition_data.get('competition', 'Unknown')}")


# ---------------------------
# Main processing
# ---------------------------
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

# ---------------------------
# LLM parsing
# ---------------------------
prompt = f"""
Extract all case competitions from this text.

Return ONLY valid JSON array.

Fields:
- competition
- month
- registration_deadline
- website
- event_type

OCR Text:
{all_text}
"""

response = llm.invoke(prompt)
ai_output = response.content.strip()

print("\n===== AI RAW OUTPUT =====\n")
print(ai_output)

# ---------------------------
# Parse JSON and save each competition
# ---------------------------
try:
    competitions = extract_json_from_llm_output(ai_output)

    if len(competitions) == 0:
        print("⚠️ No competitions found in the AI output.")
    else:
        for comp in competitions:
            # Ensure required fields exist (fill missing with empty string)
            required = ["competition", "month", "registration_deadline", "website", "event_type"]
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
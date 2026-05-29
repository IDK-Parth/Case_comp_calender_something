import os

from services.ocr_service import extract_text
from services.llm_parser import llm
from services.ocr_service import extract_text

UPLOAD_FOLDER = "uploads"

all_text = ""

# Read all images from uploads folder
for file_name in os.listdir(UPLOAD_FOLDER):

    if file_name.endswith((".png", ".jpg", ".jpeg")):

        image_path = os.path.join(UPLOAD_FOLDER, file_name)

        print(f"\nProcessing: {file_name}")

        extracted_text = extract_text(image_path)

        all_text += "\n" + extracted_text


print("\n===== OCR TEXT =====\n")
print(all_text)


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

print("\n===== AI OUTPUT =====\n")
print(response.content)
import os

# Define your custom cache path (use raw string for Windows paths)
MY_CACHE_DIR = r"E:\lang shit\case_compeletition_calender\paddle_cache"

# Set the correct environment variables for PaddleOCR 3.4.0
os.environ['PADDLE_OCR_BASE_DIR'] = MY_CACHE_DIR  # Primary variable for PaddleOCR 3.x
os.environ['PADDLEOCR_HOME'] = MY_CACHE_DIR      # Fallback for other components

# Disable the oneDNN (MKLDNN) library to avoid the NotImplementedError
os.environ['FLAGS_use_mkldnn'] = '0'

from paddleocr import PaddleOCR

# Initialize PaddleOCR
# Add show_log=False to suppress some of the verbose output if desired
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def extract_text(image_path):
    result = ocr.ocr(image_path)
    
    # The result is a list of lists. We need to handle cases where there might be no text.
    if not result or not result[0]:
        return ""
    
    extracted = []
    for line in result[0]:
        # Each line contains: [[box coordinates], (text, confidence)]
        text = line[1][0]
        extracted.append(text)
    
    return "\n".join(extracted)
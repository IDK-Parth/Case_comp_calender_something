import easyocr

# Define your custom cache path (use raw string for Windows paths)
MY_CACHE_DIR = "/easyocr_cache/"

# Initialize EasyOCR reader with custom model storage location
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=MY_CACHE_DIR)

def extract_text(image_path):
    """Extract text from an image using EasyOCR."""
    result = reader.readtext(image_path)
    
    if not result:
        return ""
    
    extracted = [item[1] for item in result]
    return "\n".join(extracted)
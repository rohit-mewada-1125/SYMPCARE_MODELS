import pytesseract
from pdf2image import convert_from_path
import os

# Set up paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

def extract_text_from_report(file_path):
    text = ""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".pdf":
        print("[INFO] Processing PDF report...")
        pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        for page_num, page in enumerate(pages, start=1):
            print(f"[INFO] Extracting text from page {page_num}...")
            text += pytesseract.image_to_string(page)
    else:
        print("[INFO] Processing image report...")
        text = pytesseract.image_to_string(file_path)
    print("text extracted successfully")
    return text

if __name__ == "__main__":
    file_path = "blood.pdf"  
    extracted_text = extract_text_from_report(file_path)
    print("\n[INFO] Extracted Text:\n", extracted_text)

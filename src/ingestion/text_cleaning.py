import re

def clean_text(text: str) -> str:
    """
    Light but safe cleaning for OCR text.
    Preserves Tamil + English + legal structure.
    """

    if not text:
        return ""

    # Normalize newlines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove excessive spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Remove page number patterns
    text = re.sub(r'Page\s+\d+\s*(of\s+\d+)?', '', text, flags=re.IGNORECASE)

    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Normalize common OCR artifacts
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')

    # Trim each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()
import os
import json
from tqdm import tqdm
from config import OCR_OUTPUT_DIR, PREPROCESS_DIR
from text_cleaning import clean_text


def preprocess_ocr_json():
    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    files = [f for f in os.listdir(OCR_OUTPUT_DIR) if f.endswith(".json")]

    print(f"Found {len(files)} OCR JSON files")

    for file in tqdm(files, desc="Preprocessing"):
        input_path = os.path.join(OCR_OUTPUT_DIR, file)
        output_path = os.path.join(
            PREPROCESS_DIR,
            file.replace(".json", "_clean.json")
        )

        # Skip if already cleaned
        if os.path.exists(output_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_pages = []
        for page in data.get("pages", []):
            cleaned_pages.append({
                "page_no": page["page_no"],
                "text": clean_text(page.get("text", ""))
            })

        cleaned_data = {
            "source_pdf": data.get("source_pdf"),
            "department_folder": data.get("department_folder"),
            "pages": cleaned_pages
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    preprocess_ocr_json()

import os
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from go_rag.config import OCR_DIR, LOG_DIR
from go_rag.config import (
    TESSERACT_PATH,
    POPPLER_PATH,
    MASTER_GO_FOLDER,
    OCR_OUTPUT_DIR,
    LOG_DIR
)

# Safety settings
Image.MAX_IMAGE_PIXELS = None
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

MAX_PIXELS = 200_000_000  # guard for huge scanned pages


def ocr_pdf(pdf_path, department_folder):
    pdf_name = os.path.basename(pdf_path)
    output_file = pdf_name.replace(".pdf", ".json")
    output_path = os.path.join(OCR_OUTPUT_DIR, output_file)

    # Skip if already processed
    if os.path.exists(output_path):
        try:
            if os.path.getsize(output_path) < 1000:
                raise ValueError("JSON too small")

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "pages" in data and len(data["pages"]) > 0:
                return "skipped"
            else:
                raise ValueError("No pages in JSON")

        except Exception:
            # corrupted or incomplete JSON → reprocess
            pass


    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    ocr_result = {
        "source_pdf": pdf_name,
        "department_folder": department_folder,
        "pages": []
    }

    for page_no in range(1, num_pages + 1):
        try:
            images = convert_from_path(
                pdf_path,
                dpi=250,
                first_page=page_no,
                last_page=page_no,
                poppler_path=POPPLER_PATH
            )

            page = images[0]

            # Skip pathological pages
            if page.size[0] * page.size[1] > MAX_PIXELS:
                print(f"⚠️ Skipping huge page {page_no} in {pdf_name}")
                del page
                del images
                continue

            text = pytesseract.image_to_string(
                page,
                lang="eng+tam",
                config="--psm 6"
            )

            ocr_result["pages"].append({
                "page_no": page_no,
                "text": text.strip()
            })

            # Hard memory cleanup
            del page
            del images

        except Exception as e:
            with open(os.path.join(LOG_DIR, "ocr_errors.log"), "a", encoding="utf-8") as log:
                log.write(f"{pdf_path} | page {page_no} | {str(e)}\n")

    # Write JSON once per PDF
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, ensure_ascii=False, indent=2)

    return "done"


def batch_ocr():
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    total = 0
    done = 0
    skipped = 0

    for root, _, files in os.walk(MASTER_GO_FOLDER):
        department = os.path.basename(root)

        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            total += 1
            pdf_path = os.path.join(root, file)

            try:
                status = ocr_pdf(pdf_path, department)
                if status == "done":
                    done += 1
                else:
                    skipped += 1

            except Exception as e:
                with open(os.path.join(LOG_DIR, "ocr_errors.log"), "a", encoding="utf-8") as log:
                    log.write(f"{pdf_path} | PDF FAILED | {str(e)}\n")

    print("\n==== OCR SUMMARY ====")
    print(f"Total PDFs found : {total}")
    print(f"OCR completed    : {done}")
    print(f"Skipped (exists) : {skipped}")
    print(f"Errors logged in : {os.path.join(LOG_DIR, 'ocr_errors.log')}")


if __name__ == "__main__":
    batch_ocr()


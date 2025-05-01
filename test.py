import os
import re
import json
from pathlib import Path

# OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Document parsing
import docx

# Pipeline configuration
data_dir = Path("./input_files")  # Folder containing PDFs, images, Word docs
output_json = Path("./extracted_qas.json")

# Regex pattern to detect question numbers like '1', '1a', '1b', etc.
QUESTION_PATTERN = re.compile(r"^(?P<qno>\d+[a-zA-Z]?)[\.\)\s]+(?P<text>.*)$")


def extract_text_from_pdf(pdf_path):
    """
    Convert each page of PDF to image and run OCR, returning concatenated text.
    """
    images = convert_from_path(str(pdf_path))
    page_texts = []
    for img in images:
        text = pytesseract.image_to_string(img)
        page_texts.append(text)
    return "\n".join(page_texts)


def extract_text_from_image(img_path):
    """
    Run OCR on a single image file.
    """
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)


def extract_text_from_word(docx_path):
    """
    Extract text from a .docx file.
    """
    doc = docx.Document(str(docx_path))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def split_into_questions(full_text):
    """
    Split full text by detected question numbers and return list of dicts with 'qno' and 'text'.
    """
    qas = []
    current = None
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = QUESTION_PATTERN.match(line)
        if m:
            # Start of a new question
            if current:
                qas.append(current)
            current = {"qno": m.group('qno'), "text": m.group('text')}  # new question block
        else:
            # Continuation of current question text
            if current:
                current['text'] += ' ' + line
            else:
                # Header text before first Q: ignore or log
                continue
    # append last block
    if current:
        qas.append(current)
    return qas


def process_file(filepath):
    """
    Determine file type and extract text accordingly.
    """
    suffix = filepath.suffix.lower()
    if suffix == '.pdf':
        return extract_text_from_pdf(filepath)
    elif suffix in ['.png', '.jpg', '.jpeg', '.tiff']:
        return extract_text_from_image(filepath)
    elif suffix in ['.docx', '.doc']:
        return extract_text_from_word(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def main():
    all_qas = []
    for file in data_dir.iterdir():
        if file.is_file():
            try:
                text = process_file(file)
                qas = split_into_questions(text)
                for qa in qas:
                    qa['source_file'] = file.name
                    all_qas.append(qa)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

    # write JSON output
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(all_qas)} question blocks across files. Output -> {output_json}")


if __name__ == '__main__':
    main()

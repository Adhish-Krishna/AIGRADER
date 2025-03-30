# AI Answer Sheet Grader

This project is a web application that automatically grades answer sheets using image recognition and NLP techniques.

## Features

- Upload answer sheets as images
- Automatic text extraction using free OCR options:
  - EasyOCR: Good with handwritten text
  - Tesseract OCR: Well-established, works great with printed text
- Answer grading using cosine similarity and NLP techniques
- Admin panel for managing answer keys and reviewing submissions
- Score visualization and detailed feedback

## Setup

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system (optional)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR (optional, if you want to use Tesseract):
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Copy `.env.sample` to `.env` and update as needed:
   ```
   # Select your preferred OCR engine ("easyocr", "tesseract", "combined")
   OCR_ENGINE=easyocr
   
   # Path to tesseract executable (if not in PATH)
   TESSERACT_CMD=/path/to/tesseract
   ```

### Running the application

1. Start the application:
   ```
   python main.py
   ```
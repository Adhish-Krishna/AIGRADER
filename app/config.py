"""
Configuration settings for the AI Answer Sheet Grader application.
"""
import os
import platform
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATE_DIR = APP_DIR / "templates"
DATABASE_DIR = APP_DIR / "database"

# Database files
UPLOADS_DIR = DATABASE_DIR / "files"
ANSWER_KEYS_DIR = DATABASE_DIR / "answer_keys"
SUBMISSIONS_FILE = DATABASE_DIR / "submissions.json"
ANSWER_KEYS_FILE = DATABASE_DIR / "answer_keys.json"

# Ensure all directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ANSWER_KEYS_DIR.mkdir(parents=True, exist_ok=True)

# API settings
API_TITLE = "AI Answer Sheet Grader"
API_VERSION = "1.0.0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# File upload settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "app/database/files")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))  # 5MB default

# OCR settings
OCR_ENGINE = "pytesseract"
# Set tesseract path based on platform
if platform.system() == "Windows":
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
else:
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Default OCR configurations
OCR_DEFAULT_CONFIG = os.getenv("OCR_DEFAULT_CONFIG", "--oem 3 --psm 6")
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")

# Debug settings for OCR
OCR_DEBUG = os.getenv("OCR_DEBUG", "False").lower() in ("true", "1", "t")
OCR_DEBUG_DIR = os.getenv("OCR_DEBUG_DIR", "app/debug/ocr")
if OCR_DEBUG:
    Path(OCR_DEBUG_DIR).mkdir(parents=True, exist_ok=True)

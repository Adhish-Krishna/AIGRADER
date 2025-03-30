import os
import logging
import uuid
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
import time

from fastapi import UploadFile

from app.config import UPLOADS_DIR, MAX_UPLOAD_SIZE

logger = logging.getLogger(__name__)

def get_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate a unique filename with timestamp to prevent overwriting.
    
    Args:
        original_filename: The original filename
        prefix: Optional prefix for the filename
        
    Returns:
        A unique filename with timestamp
    """
    filename, ext = os.path.splitext(original_filename)
    timestamp = int(time.time())
    
    if prefix:
        return f"{prefix}_{timestamp}{ext}"
    else:
        return f"{filename}_{timestamp}{ext}"

async def save_upload_file(file: UploadFile, directory: Path, prefix: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Save an uploaded file with proper validation and error handling.
    
    Args:
        file: The uploaded file
        directory: Directory to save the file in
        prefix: Optional prefix for the saved filename
        
    Returns:
        Tuple of (success, saved_path, error_message)
    """
    try:
        # Ensure upload directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Validate file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Save file with unique name
        filename = get_unique_filename(file.filename, prefix)
        file_path = os.path.join(directory, filename)
        
        # Write file in chunks to handle large files
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE:
                    # Delete partially written file
                    os.remove(file_path)
                    return False, None, f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE / (1024*1024)}MB"
                f.write(chunk)
        
        return True, file_path, None
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return False, None, str(e)

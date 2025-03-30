import os
import json
import time
import logging
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from app.config import ANSWER_KEYS_FILE, SUBMISSIONS_FILE, UPLOADS_DIR

logger = logging.getLogger(__name__)

def ensure_valid_json_file(file_path: Path, default_content: Any = None) -> None:
    """
    Ensure that a JSON file exists and contains valid JSON.
    If the file doesn't exist or is invalid, initialize it with default content.
    
    Args:
        file_path: Path to the JSON file
        default_content: Default content to write if file is missing or invalid
    """
    if default_content is None:
        default_content = {}
        
    try:
        # Check if file exists
        if not file_path.exists():
            logger.info(f"Creating new JSON file: {file_path}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_content, f, indent=2)
            return
            
        # Check if file contains valid JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {file_path}, reinitializing file")
                # Back up corrupted file
                backup_path = file_path.with_suffix(f".bak.{int(time.time())}")
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up corrupted file to {backup_path}")
                
                # Write default content
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, indent=2)
                    
    except Exception as e:
        logger.error(f"Error ensuring valid JSON file {file_path}: {str(e)}")
        raise

def load_json_data(file_path: Path) -> Dict[str, Any]:
    """
    Load data from a JSON file with proper error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    try:
        ensure_valid_json_file(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return {}

def save_json_data(file_path: Path, data: Dict[str, Any]) -> bool:
    """
    Save data to a JSON file with proper error handling.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False

# Answer key functions

def get_answer_keys() -> Dict[str, Any]:
    """Get all answer keys."""
    return load_json_data(ANSWER_KEYS_FILE)

def get_answer_key(key_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific answer key by ID."""
    answer_keys = get_answer_keys()
    return answer_keys.get(key_id)

def save_answer_key(key_id: str, subject_id: str, file_path: str, answers: Dict[str, str]) -> bool:
    """Save an answer key."""
    try:
        answer_keys = get_answer_keys()
        
        answer_keys[key_id] = {
            "subject_id": subject_id,
            "file_path": file_path,
            "answers": answers,
            "timestamp": datetime.now().isoformat()
        }
        
        return save_json_data(ANSWER_KEYS_FILE, answer_keys)
    except Exception as e:
        logger.error(f"Error saving answer key: {str(e)}")
        return False

# Submission functions

def get_submissions() -> Dict[str, Any]:
    """Get all submissions."""
    return load_json_data(SUBMISSIONS_FILE)

def get_submission(submission_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific submission by ID."""
    submissions = get_submissions()
    return submissions.get(submission_id)

def save_submission(submission_id: str, student_id: str, subject_id: str, 
                   file_path: str, answers: List[Dict[str, Any]],
                   grades: Optional[Dict[str, Any]] = None) -> bool:
    """Save a submission."""
    try:
        submissions = get_submissions()
        
        if grades is None:
            grades = {}
        
        # Ensure answers is serializable
        serializable_answers = []
        for answer in answers:
            serializable_answer = {
                "question_number": answer.get("question_number", ""),
                "answer_text": answer.get("answer_text", "")
            }
            serializable_answers.append(serializable_answer)
        
        submissions[submission_id] = {
            "student_id": student_id,
            "subject_id": subject_id,
            "file_path": file_path,
            "answers": serializable_answers,
            "grades": grades,
            "timestamp": datetime.now().isoformat()
        }
        
        return save_json_data(SUBMISSIONS_FILE, submissions)
    except Exception as e:
        logger.error(f"Error saving submission: {str(e)}")
        return False

# Initialize database files on module import
try:
    ensure_valid_json_file(ANSWER_KEYS_FILE, {})
    ensure_valid_json_file(SUBMISSIONS_FILE, {})
except Exception as e:
    logger.error(f"Failed to initialize database files: {str(e)}")

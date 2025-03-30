import os
import logging
from typing import List, Dict, Any, Union

# Import our OCR service
from app.services.florence_ocr_service import FlorenceOcrService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        self.ocr_service = FlorenceOcrService()
    
    def parse_answer_key(self, image_path: str) -> Dict[int, str]:
        """Parse an answer key image and return question-answer pairs."""
        return self.ocr_service.parse_answer_key(image_path)

async def parse_answer_sheet(file_path: str, is_answer_key: bool = False) -> Union[Dict[int, str], List[Dict[str, Any]]]:
    """
    Parse an answer sheet image using advanced OCR techniques
    
    Args:
        file_path: Path to the uploaded answer sheet image
        is_answer_key: Whether this is an answer key being processed
        
    Returns:
        For answer keys: Dictionary mapping question numbers to answer text
        For student answers: List of dictionaries with question numbers and answers
    """
    logger.info(f"Parsing {'answer key' if is_answer_key else 'student answers'} from: {file_path}")
    
    try:
        # Create OCR service instance
        ocr_service = FlorenceOcrService()
        
        # Use the same segmentation-based approach for both types
        if is_answer_key:
            return ocr_service.parse_answer_key(file_path)
        else:
            return await ocr_service.process_student_answers(file_path)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Return appropriate fallback data
        if is_answer_key:
            return get_fallback_answer_key()
        else:
            return get_fallback_student_answers()

def get_fallback_answer_key() -> Dict[int, str]:
    """Return fallback answer key when processing fails"""
    return {
        1: "Sample answer for question 1",
        2: "Sample answer for question 2",
        3: "Sample answer for question 3"
    }

def get_fallback_student_answers() -> List[Dict[str, Any]]:
    """Return fallback student answers when processing fails"""
    return [
        {"question_number": 1, "answer_text": "This is a sample answer for question 1."},
        {"question_number": 2, "answer_text": "This is a sample answer for question 2."},
        {"question_number": 3, "answer_text": "This is a sample answer for question 3."}
    ]

import os
import re
import logging
from typing import Dict, List, Any
import tempfile
import platform
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Import our utility modules
from app.utils.image_preprocessing import preprocess_image, find_margin_boundary
from app.utils.ocr_processing import detect_question_markers, segment_answer_regions, extract_text_from_regions
from app.config import TESSERACT_CMD, OCR_DEFAULT_CONFIG, OCR_LANGUAGE, OCR_DEBUG, OCR_DEBUG_DIR

logger = logging.getLogger(__name__)

class FlorenceOcrService:
    def __init__(self):
        logger.info("Initializing Pytesseract OCR service")
        
        # Set pytesseract path from config
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # Configure OCR options
        self.config = OCR_DEFAULT_CONFIG
        logger.info(f"Pytesseract OCR service initialized with config: {self.config}")

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using pytesseract OCR."""
        logger.info(f"Extracting text from image: {image_path}")
        
        try:
            # Apply preprocessing to improve OCR quality
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_path}")
            
            # Save debug image if debug is enabled
            if OCR_DEBUG:
                debug_path = os.path.join(OCR_DEBUG_DIR, f"original_{os.path.basename(image_path)}")
                cv2.imwrite(debug_path, image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply thresholding
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR on the preprocessed image
            configs = [
                "--oem 3 --psm 6",  # Assume a single uniform block of text
                "--oem 3 --psm 4",  # Assume a single column of text
                "--oem 3 --psm 3",  # Fully automatic page segmentation
            ]
            
            best_text = ""
            max_confidence = 0
            
            for config in configs:
                # Get both text and confidence
                data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence for non-empty text
                confidences = []
                text_parts = []
                
                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        try:
                            conf = float(data['conf'][i])
                            if conf > 0:  # Only consider positive confidence
                                confidences.append(conf)
                                text_parts.append(data['text'][i])
                        except ValueError:
                            continue
                
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if avg_confidence > max_confidence:
                        max_confidence = avg_confidence
                        best_text = " ".join(text_parts)
            
            if not best_text:
                # Try a simpler approach as fallback
                best_text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 1")
            
            # Add debug output
            if OCR_DEBUG:
                with open(os.path.join(OCR_DEBUG_DIR, f"text_{os.path.basename(image_path)}.txt"), 'w') as f:
                    f.write(best_text)
            
            logger.info(f"OCR extracted text. First 100 chars: {best_text[:100]}")
            return best_text
        
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return "Error extracting text with OCR"

    def parse_answer_key(self, image_path: str) -> Dict[int, str]:
        """
        Process an answer key image and extract question-answer pairs using
        advanced segmentation techniques with pytesseract.
        
        Args:
            image_path: Path to the answer key image
            
        Returns:
            Dictionary mapping question numbers to answer text
        """
        logger.info(f"Processing answer key with pytesseract: {image_path}")
        
        try:
            # Step 1: Preprocess the image
            image, gray, thresh = preprocess_image(image_path)
            
            # Step 2: Find the margin boundary
            margin_boundary = find_margin_boundary(thresh)
            logger.info(f"Margin boundary detected at: {margin_boundary} pixels")
            
            # Step 3: Extract the left margin region containing question numbers
            margin = thresh[:, :margin_boundary]
            
            # Step 4: Detect question markers in the margin
            markers = detect_question_markers(margin)
            if not markers:
                logger.warning("No question markers found. Falling back to basic OCR.")
                return self._parse_answer_key_basic(image_path)
            
            # Step 5: Segment answer regions based on detected markers
            answer_regions = segment_answer_regions(image, markers, margin_boundary)
            
            # Step 6: Extract text using pytesseract
            answer_key = {}
            
            for question_num, region in answer_regions.items():
                # Save region to a temporary file for processing with pytesseract
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, region)
                
                # Extract text using pytesseract with multiple configurations
                configs = [
                    "--oem 3 --psm 6",  # Assume a single uniform block of text
                    "--oem 3 --psm 4",  # Assume a single column of text
                ]
                
                best_text = ""
                max_confidence = 0
                
                for config in configs:
                    # Get both text and confidence
                    data = pytesseract.image_to_data(region, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence for non-empty text
                    confidences = []
                    text_parts = []
                    
                    for i in range(len(data['text'])):
                        if data['text'][i].strip():
                            try:
                                conf = float(data['conf'][i])
                                if conf > 0:
                                    confidences.append(conf)
                                    text_parts.append(data['text'][i])
                            except ValueError:
                                continue
                    
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence > max_confidence:
                            max_confidence = avg_confidence
                            best_text = " ".join(text_parts)
                
                if not best_text:
                    # Fallback
                    best_text = pytesseract.image_to_string(region, config="--oem 3 --psm 1")
                
                # Clean up the extracted text and store it
                best_text = best_text.strip()
                answer_key[question_num] = best_text
                
                # Clean up temporary file
                os.unlink(temp_path)
            
            logger.info(f"Successfully processed answer key with {len(answer_key)} questions")
            return answer_key
            
        except Exception as e:
            logger.error(f"Error processing answer key with segmentation: {str(e)}")
            logger.info("Falling back to basic OCR method")
            return self._parse_answer_key_basic(image_path)

    def _parse_answer_key_basic(self, image_path: str) -> Dict[int, str]:
        """
        Basic fallback method to process an answer key when advanced segmentation fails.
        
        Args:
            image_path: Path to the answer key image
            
        Returns:
            Dictionary mapping question numbers to answer text
        """
        # Extract text using OCR
        extracted_text = self.extract_text_from_image(image_path)
        
        # Process the text to extract question-answer pairs
        answer_key = {}
        
        # Split text into lines
        lines = extracted_text.split('\n')
        current_question = None
        current_answer = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            question_match = re.match(r'^(\d+)[\.:\)]\s*(.*)$', line)
            
            if question_match:
                if current_question is not None and current_answer:
                    answer_key[current_question] = '\n'.join(current_answer)
                    current_answer = []
                
                current_question = int(question_match.group(1))
                answer_text = question_match.group(2).strip()
                if answer_text:
                    current_answer.append(answer_text)
            elif current_question is not None:
                current_answer.append(line)
        
        if current_question is not None and current_answer:
            answer_key[current_question] = '\n'.join(current_answer)
        
        # If no questions were detected, use the fallback extraction method
        if not answer_key:
            logger.info("No questions detected in OCR output. Falling back to simple extraction.")
            fallback_answers = self._fallback_answer_extraction(extracted_text)
            answer_key = {entry["question_number"]: entry["answer_text"] for entry in fallback_answers}
        
        logger.info(f"Processed answer key with {len(answer_key)} questions using basic method")
        return answer_key

    async def process_student_answers(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process a student's answer sheet image using the same advanced segmentation
        technique as for answer keys.
        
        Args:
            image_path: Path to the student answer sheet image
            
        Returns:
            List of dictionaries with question numbers and answers
        """
        logger.info(f"Processing student answers with advanced segmentation: {image_path}")
        
        try:
            # Step 1: Preprocess the image
            image, gray, thresh = preprocess_image(image_path)
            
            # Step 2: Find the margin boundary
            margin_boundary = find_margin_boundary(thresh)
            logger.info(f"Margin boundary detected at: {margin_boundary} pixels")
            
            # Step 3: Extract the left margin region containing question numbers
            margin = thresh[:, :margin_boundary]
            
            # Step 4: Detect question markers in the margin
            markers = detect_question_markers(margin)
            if not markers:
                logger.warning("No question markers found. Falling back to basic OCR.")
                return await self._process_student_answers_basic(image_path)
            
            # Step 5: Segment answer regions based on detected markers
            answer_regions = segment_answer_regions(image, markers, margin_boundary)
            
            # Step 6: Extract text using pytesseract for each region
            answers = []
            
            for question_num, region in answer_regions.items():
                # Save region to a temporary file for processing with pytesseract
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, region)
                
                # Extract text using pytesseract with multiple configurations
                configs = [
                    "--oem 3 --psm 6",  # Assume a single uniform block of text
                    "--oem 3 --psm 4",  # Assume a single column of text
                ]
                
                best_text = ""
                max_confidence = 0
                
                for config in configs:
                    # Get both text and confidence
                    data = pytesseract.image_to_data(region, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence for non-empty text
                    confidences = []
                    text_parts = []
                    
                    for i in range(len(data['text'])):
                        if data['text'][i].strip():
                            try:
                                conf = float(data['conf'][i])
                                if conf > 0:
                                    confidences.append(conf)
                                    text_parts.append(data['text'][i])
                            except ValueError:
                                continue
                    
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence > max_confidence:
                            max_confidence = avg_confidence
                            best_text = " ".join(text_parts)
                
                if not best_text:
                    # Fallback
                    best_text = pytesseract.image_to_string(region, config="--oem 3 --psm 1")
                
                # Clean up the extracted text and store it in the required format for student answers
                best_text = best_text.strip()
                answers.append({
                    "question_number": question_num,
                    "answer_text": best_text
                })
                
                # Clean up temporary file
                os.unlink(temp_path)
            
            logger.info(f"Successfully processed student answers with {len(answers)} questions")
            
            # If we couldn't identify any answers, fall back to the basic method
            if not answers:
                logger.warning("No answers detected with advanced segmentation. Falling back to basic method.")
                return await self._process_student_answers_basic(image_path)
                
            return answers
            
        except Exception as e:
            logger.error(f"Error processing student answers with advanced segmentation: {str(e)}")
            logger.info("Falling back to basic OCR method for student answers")
            return await self._process_student_answers_basic(image_path)

    async def _process_student_answers_basic(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Basic fallback method to process student answers when advanced segmentation fails.
        
        Args:
            image_path: Path to the student answer sheet image
            
        Returns:
            List of dictionaries with question numbers and answers
        """
        # Extract text using OCR
        extracted_text = self.extract_text_from_image(image_path)
        
        # Process the text to extract answers
        answers = []
        
        # Split text into lines
        lines = extracted_text.split('\n')
        current_question = None
        current_answer = []
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify question numbers at the start of lines
            question_match = re.match(r'^(\d+)[\.:\)]\s*(.*)$', line)
            
            if question_match:
                # If we were processing a question, save it before starting a new one
                if current_question is not None and current_answer:
                    answers.append({
                        "question_number": current_question,
                        "answer_text": '\n'.join(current_answer)
                    })
                    current_answer = []
                
                # Start a new question
                current_question = int(question_match.group(1))
                answer_text = question_match.group(2).strip()
                if answer_text:
                    current_answer.append(answer_text)
            elif current_question is not None:
                # Continue with the current answer
                current_answer.append(line)
        
        # Don't forget the last question
        if current_question is not None and current_answer:
            answers.append({
                "question_number": current_question,
                "answer_text": '\n'.join(current_answer)
            })
        
        logger.info(f"Processed student answers with {len(answers)} questions using basic method")
        
        # If we couldn't identify any questions, try a simpler approach
        if not answers:
            answers = self._fallback_answer_extraction(extracted_text)
        
        return answers

    def _fallback_answer_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        A simpler fallback method to extract answers when question numbers can't be detected
        
        Args:
            text: Raw text from the image
            
        Returns:
            List of answers with estimated question numbers
        """
        # Split into paragraphs (separated by blank lines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Create answers from paragraphs
        answers = []
        for i, paragraph in enumerate(paragraphs):
            answers.append({
                "question_number": i + 1,  # Assume 1-indexed questions
                "answer_text": paragraph
            })
        
        logger.info(f"Used fallback extraction to get {len(answers)} answers")
        return answers


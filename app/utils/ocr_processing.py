import cv2
import numpy as np
import pytesseract
import logging
import re
import platform
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Configure pytesseract path for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def cluster_markers(markers, y_threshold=30):
    """
    Group markers that are closer than y_threshold pixels.
    
    Args:
        markers: List of detected marker dictionaries
        y_threshold: Maximum vertical distance to consider markers as the same
        
    Returns:
        List of merged markers
    """
    if not markers:
        return []
        
    markers.sort(key=lambda m: m['y'])
    clustered = [markers[0]]
    
    for marker in markers[1:]:
        last = clustered[-1]
        if abs(marker['y'] - last['y']) < y_threshold:
            last['y'] = (last['y'] + marker['y']) // 2
            last['height'] = max(last.get('height', 20), marker.get('height', 20))
        else:
            clustered.append(marker)
            
    return clustered

def detect_question_markers(margin_image):
    """
    Detect question number markers in the margin area using pytesseract.
    
    Args:
        margin_image: Grayscale or thresholded image of the margin area
        
    Returns:
        List of dictionaries containing question markers with their positions
    """
    h, w = margin_image.shape
    logger.info(f"Detecting question markers in margin area of shape {h}x{w}")
    
    # Try OCR detection first
    markers = []
    configurations = [
        {'psm': 10, 'threshold': 30, 'scale': 3},  # Single character
        {'psm': 6, 'threshold': 25, 'scale': 3},   # Block of text
        {'psm': 7, 'threshold': 20, 'scale': 4},   # Single line
    ]
    
    for config in configurations:
        psm = config['psm']
        confidence_threshold = config['threshold']
        scale_factor = config['scale']
        custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
        
        working_margin = margin_image.copy()
        kernel = np.ones((2, 2), np.uint8)
        working_margin = cv2.dilate(working_margin, kernel, iterations=1)
        
        margin_scaled = cv2.resize(working_margin, None, 
                                   fx=scale_factor, 
                                   fy=scale_factor,
                                   interpolation=cv2.INTER_CUBIC)
        
        details = pytesseract.image_to_data(margin_scaled, 
                                            output_type=pytesseract.Output.DICT, 
                                            config=custom_config)
        
        config_markers = []
        for i in range(len(details['text'])):
            text = details['text'][i].strip()
            try:
                conf = float(details['conf'][i])
            except ValueError:
                conf = 0.0
            if text.isdigit() and conf > confidence_threshold:
                config_markers.append({
                    'number': int(text),
                    'y': int(details['top'][i] / scale_factor),
                    'height': int(details['height'][i] / scale_factor),
                    'confidence': conf,
                    'x': int(details['left'][i] / scale_factor),
                    'width': int(details['width'][i] / scale_factor)
                })
        if config_markers:
            logger.info(f"Found {len(config_markers)} markers using PSM {psm}")
            markers.extend(config_markers)
    
    # If OCR fails, use contour detection as fallback
    if not markers:
        logger.warning("OCR detection failed. Using contour-based detection...")
        
        # Use contour detection to find potential digit areas
        contours, _ = cv2.findContours(margin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 100
        max_area = 1500
        expected_digits = range(1, 21)  # Assume up to 20 questions
        
        # Sort contours by vertical position
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Filter contours by size and aspect ratio
            if min_area < area < max_area and 0.25 < aspect_ratio < 2.0:
                if i < len(expected_digits):
                    number = expected_digits[i]
                    markers.append({
                        'number': number,
                        'y': y,
                        'height': h,
                        'x': x,
                        'width': w,
                        'confidence': 30  # Arbitrary confidence for contour detection
                    })
    
    # If we still couldn't detect markers, generate some evenly spaced ones
    if not markers:
        logger.warning("No question markers detected. Generating evenly spaced markers.")
        num_questions = 10
        step = h // (num_questions + 1)
        
        for i in range(1, num_questions + 1):
            markers.append({
                'number': i,
                'y': i * step,
                'height': 20,
                'x': 10,
                'width': 20
            })
    
    # Cluster markers that may be too close
    markers = cluster_markers(markers)
    logger.info(f"Detected {len(markers)} question markers")
    
    return markers

def segment_answer_regions(image, markers, margin_boundary):
    """
    Segment answer regions based on detected markers.
    Each region spans from one marker's y coordinate to the next.
    
    Args:
        image: Original or preprocessed image
        markers: List of detected question markers
        margin_boundary: X-coordinate separating margin from answer area
        
    Returns:
        Dictionary mapping question numbers to image regions
    """
    h, w = image.shape[:2]
    answer_regions = {}
    
    # Sort markers by vertical position
    markers.sort(key=lambda m: m['y'])
    
    for i, marker in enumerate(markers):
        y_start = marker['y']
        y_end = markers[i+1]['y'] if i < len(markers)-1 else h
        
        # Ensure region has minimum height
        if y_end - y_start < 20:
            continue
            
        # Extract region, excluding margin
        if len(image.shape) == 3:  # Color image
            region = image[y_start:y_end, margin_boundary:w]
        else:  # Grayscale image
            region = image[y_start:y_end, margin_boundary:w]
            
        answer_regions[marker['number']] = region
        
    logger.info(f"Segmented {len(answer_regions)} answer regions")
    return answer_regions

def extract_text_from_regions(regions, ocr_processor=None):
    """
    Extract text from segmented answer regions using pytesseract OCR.
    
    Args:
        regions: Dictionary mapping question numbers to image regions
        ocr_processor: OCR processor instance (optional)
        
    Returns:
        Dictionary mapping question numbers to extracted text
    """
    extracted_text = {}
    
    for question_num, region in regions.items():
        # Apply additional preprocessing for better OCR results
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Try multiple OCR configurations for better accuracy
        configs = [
            '--oem 3 --psm 6',  # Assume a single uniform block of text
            '--oem 3 --psm 4',  # Assume a single column of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
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
        
        if not best_text:  # Fallback
            best_text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 1")
        
        extracted_text[question_num] = best_text.strip()
        logger.info(f"Question {question_num}: Extracted text with confidence {max_confidence:.2f}")
    
    return extracted_text

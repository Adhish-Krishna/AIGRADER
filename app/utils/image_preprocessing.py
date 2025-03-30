import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess an image for OCR by converting to grayscale, enhancing contrast,
    and applying thresholding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (original_image, grayscale_image, thresholded_image)
    """
    logger.info(f"Preprocessing image: {image_path}")
    
    try:
        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use Otsu thresholding (good for handwritten strokes)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Enhance strokes with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        logger.info(f"Image preprocessing complete: {image_path}")
        return image, gray, thresh
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def find_margin_boundary(thresh):
    """
    Find the boundary between the margin (containing question numbers)
    and the answer area.
    
    Args:
        thresh: Thresholded image
        
    Returns:
        int: X-coordinate of the margin boundary
    """
    try:
        # Compute vertical projection (sum of white pixels per column)
        vertical_projection = np.sum(thresh, axis=0)
        max_density = np.max(vertical_projection)
        
        # Find first column where density drops below 50% of maximum
        margin_boundary = int(np.argmax(vertical_projection < (max_density * 0.5)))
        
        if margin_boundary == 0:
            logger.warning("Margin boundary not found; using default value.")
            margin_boundary = 300
            
        margin_boundary = max(margin_boundary, 30)
        # Ensure margin does not exceed 10% of image width
        margin_boundary = min(margin_boundary, int(0.1 * thresh.shape[1]))
        
        logger.info(f"Margin boundary detected at: {margin_boundary} pixels")
        return margin_boundary
        
    except Exception as e:
        logger.error(f"Error finding margin boundary: {str(e)}")
        return 50  # Default fallback value

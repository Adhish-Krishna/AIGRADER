"""
Test script for OCR functionality.
Run this script to test OCR on a sample image.
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add the parent directory to path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.florence_ocr_service import FlorenceOcrService
from app.utils.image_preprocessing import preprocess_image, find_margin_boundary
from app.utils.ocr_processing import detect_question_markers, segment_answer_regions

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ocr(image_path):
    """Run OCR on an image and print the results."""
    try:
        ocr_service = FlorenceOcrService()
        
        # Basic OCR test
        logger.info(f"Testing basic OCR on {image_path}")
        text = ocr_service.extract_text_from_image(image_path)
        logger.info(f"Extracted text:\n{text}")
        
        # Advanced segmentation test
        logger.info(f"Testing advanced segmentation on {image_path}")
        answers = ocr_service.parse_answer_key(image_path)
        logger.info(f"Extracted answers:\n{json.dumps(answers, indent=2)}")
        
        # Save the results
        output_dir = Path('ocr_test_results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f"{Path(image_path).stem}_text.txt", 'w') as f:
            f.write(text)
            
        with open(output_dir / f"{Path(image_path).stem}_answers.json", 'w') as f:
            json.dump(answers, f, indent=2)
            
        logger.info(f"Results saved to {output_dir}")
        
        return answers
    
    except Exception as e:
        logger.error(f"Error testing OCR: {str(e)}")
        return None

def visualize_segmentation(image_path):
    """Visualize the segmentation process."""
    try:
        # Preprocess the image
        image, gray, thresh = preprocess_image(image_path)
        
        # Find margin boundary
        margin_boundary = find_margin_boundary(thresh)
        logger.info(f"Margin boundary at {margin_boundary} pixels")
        
        # Extract margin and detect markers
        margin = thresh[:, :margin_boundary]
        markers = detect_question_markers(margin)
        logger.info(f"Detected {len(markers)} question markers")
        
        # Segment answer regions
        regions = segment_answer_regions(image, markers, margin_boundary)
        logger.info(f"Segmented {len(regions)} answer regions")
        
        # Visualize the results
        output_dir = Path('ocr_viz_results')
        output_dir.mkdir(exist_ok=True)
        
        # Draw margin boundary on image
        viz_image = image.copy()
        cv2.line(viz_image, (margin_boundary, 0), 
                 (margin_boundary, image.shape[0]), (0, 0, 255), 2)
        
        # Draw markers
        for marker in markers:
            y = marker['y']
            cv2.line(viz_image, (0, y), (margin_boundary, y), (0, 255, 0), 2)
            cv2.putText(viz_image, str(marker['number']), 
                        (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_segmentation.png"), viz_image)
        
        # Save each segmented region
        for q_num, region in regions.items():
            cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_q{q_num}.png"), region)
        
        logger.info(f"Visualization saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error visualizing segmentation: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR functionality")
    parser.add_argument("image_path", help="Path to an image file to test OCR on")
    parser.add_argument("--visualize", action="store_true", help="Visualize the segmentation process")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        sys.exit(1)
    
    if args.visualize:
        visualize_segmentation(args.image_path)
    
    test_ocr(args.image_path)

import cv2 as cv
import numpy as np

def detect_horizontal_lines(image_path):
    # Load the image in grayscale
    src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Error loading image: {image_path}")
        return

    # Detect edges using Canny
    edges = cv.Canny(src, 50,150, None, 3)

    # Convert edges to a BGR image for visualization
    cdstP = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # Apply Probabilistic Hough Line Transform
    linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

    # Draw horizontal lines
    if linesP is not None:
        for i in range(len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]
            # Check if the line is approximately horizontal
            if abs(y1 - y2) < 5:  # Allow a small vertical difference
                cv.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

    # Display results
    cv.imshow("Detected Horizontal Lines", cdstP)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage
detect_horizontal_lines("13.jpg")

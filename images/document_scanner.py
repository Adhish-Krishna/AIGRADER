import cv2
import numpy as np

def enhance_document_scanner(image_path):
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (int(480 * 2), int(640 * 2)))

    # Preprocess image
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    BlurredFrame = cv2.GaussianBlur(GrayImg, (5, 5), 1)
    CannyFrame = cv2.Canny(BlurredFrame, 75, 200)

    # Detect lines using Hough Line Transformation
    lines = cv2.HoughLinesP(CannyFrame, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    LineFrame = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(LineFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find contours
    contours, _ = cv2.findContours(CannyFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ContourFrame = img.copy()
    ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (255, 0, 255), 4)

    # Find the biggest contour (assume it's the document)
    maxArea = 0
    biggest = None
    for i in contours:
        area = cv2.contourArea(i)
        if area > 500:
            peri = cv2.arcLength(i, True)
            edges = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > maxArea and len(edges) == 4:  # Look for quadrilateral shapes
                biggest = edges
                maxArea = area

    # Highlight detected corners
    CornerFrame = img.copy()
    if biggest is not None:
        biggest = biggest.reshape(4, 2)
        # Sort points for consistent order (top-left, top-right, bottom-right, bottom-left)
        rect = np.zeros((4, 2), dtype="float32")
        s = biggest.sum(axis=1)
        rect[0] = biggest[np.argmin(s)]  # Top-left
        rect[2] = biggest[np.argmax(s)]  # Bottom-right
        diff = np.diff(biggest, axis=1)
        rect[1] = biggest[np.argmin(diff)]  # Top-right
        rect[3] = biggest[np.argmax(diff)]  # Bottom-left

        # Draw corners
        for x, y in rect:
            cv2.circle(CornerFrame, (int(x), int(y)), 10, (0, 255, 0), -1)

        # Perspective transformation
        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, height))
    else:
        warped = img.copy()

    # Display results
    img = cv2.resize(img, (480, 640))
    GrayImg = cv2.resize(GrayImg, (480, 640))
    BlurredFrame = cv2.resize(BlurredFrame, (480, 640))
    CannyFrame = cv2.resize(CannyFrame, (480, 640))
    ContourFrame = cv2.resize(ContourFrame, (480, 640))
    CornerFrame = cv2.resize(CornerFrame, (480, 640))
    warped = cv2.resize(warped, (480, 640))

    cv2.imshow("Original Image", img)
    cv2.imshow("Gray Image", GrayImg)
    cv2.imshow("Blurred Frame", BlurredFrame)
    cv2.imshow("Canny Edges", CannyFrame)
    cv2.imshow("Contours", ContourFrame)
    cv2.imshow("Corners Detected", CornerFrame)
    cv2.imshow("Warped Document", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
enhance_document_scanner("14.jpg")

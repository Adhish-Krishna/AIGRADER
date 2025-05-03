import os
import re
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import docx
from typing import List, Union

# Pipeline configuration
data_dir = Path("./input_files")  # Folder containing PDFs, images, Word docs
output_json = Path("./extracted_qas.json")

# Regex for question numbers
QUESTION_PATTERN = re.compile(r"^(?P<qno>\d+[a-zA-Z]?)[\.\)\s]+")

# Utility: deskew image using Hough-based skew detection
def deskew_image(src: np.ndarray) -> np.ndarray:
    edges = cv.Canny(src, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return src
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi/2)
        angles.append(angle)
    if not angles:
        return src
    median_angle = np.median(angles)
    angle_deg = median_angle * 180 / np.pi
    # rotate
    (h, w) = src.shape
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv.warpAffine(src, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rotated

# Utility: get horizontal projection valleys
def projection_valleys(bin_img: np.ndarray) -> List[int]:
    # sum inverted binary (text=1, bg=0)
    proj = np.sum(bin_img // 255, axis=1)
    # find valleys: local minima with wide support
    valley_threshold = np.max(proj) * 0.5
    valleys = [i for i in range(1, len(proj)-1) if proj[i] < proj[i-1] and proj[i] < proj[i+1] and proj[i] < valley_threshold]
    return valleys


def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    try:
        pages = convert_from_path(pdf_path)
        images = []
        for page in pages:
            gray = cv.cvtColor(np.array(page), cv.COLOR_RGB2GRAY)
            cropped = gray[10:-10, 10:-10]
            images.append(cropped)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def extract_text_from_word(docx_path: str) -> str:
    doc = docx.Document(str(docx_path))
    return "\n".join([p.text for p in doc.paragraphs])


def split_into_questions(full_text: str) -> List[dict]:
    qas, current = [], None
    for line in full_text.splitlines():
        m = QUESTION_PATTERN.match(line.strip())
        if m:
            if current:
                qas.append(current)
            qno = m.group('qno')
            current = {"qno": qno, "text": line[m.end():].strip()}
        elif current:
            current['text'] += ' ' + line.strip()
    if current:
        qas.append(current)
    return qas


def detect_and_split(image_input: Union[str, np.ndarray], output_dir="chunks", debug=False, is_array=False) -> List[np.ndarray]:
    # Load and deskew
    src = image_input if is_array else cv.imread(image_input, cv.IMREAD_GRAYSCALE)
    if src is None:
        raise FileNotFoundError(f"Could not load {image_input}")
    src = deskew_image(src)
    h, w = src.shape
    # Binarize
    blur = cv.GaussianBlur(src, (5, 5), 0)
    bin_img = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 10)
    # Morphology for horizontals
    ker_len = max(10, w // 4)
    horiz_kernel = cv.getStructuringElement(cv.MORPH_RECT, (ker_len, 1))
    morph = cv.morphologyEx(bin_img, cv.MORPH_OPEN, horiz_kernel, iterations=2)
    # Hough for slight tilt
    edges = cv.Canny(bin_img, 50, 150, apertureSize=3)
    linesP = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=ker_len//2, maxLineGap=20)
    angle_tol = 15
    hough_mask = np.zeros_like(bin_img)
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang < angle_tol or ang > (180-angle_tol):
                cv.line(hough_mask, (x1,y1), (x2,y2), 255, 2)
    combined = cv.bitwise_or(morph, hough_mask)
    # Projection valleys
    valleys = projection_valleys(bin_img)
    # Consensus cuts: intersection of valleys & combined mask rows
    cuts = []
    for y in valleys:
        if np.mean(combined[y]) > 10:  # some line pixels present
            cuts.append(y)
    # fallback
    if not cuts:
        cuts = [h//2]
    # merge close
    cuts = sorted(cuts)
    merged = [cuts[0]]
    tol = 20
    for y in cuts[1:]:
        if abs(y - merged[-1]) < tol:
            merged[-1] = (merged[-1] + y)//2
        else:
            merged.append(y)
    # build slices
    bounds = [0] + merged + [h]
    chunks = []
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(bounds)-1):
        y0, y1 = bounds[i], bounds[i+1]
        if y1 - y0 < 20: continue
        band = src[y0:y1, :]
        chunks.append(band)
        cv.imwrite(os.path.join(output_dir, f"chunk_{i}.png"), band)
    return chunks


def process_document(file_path: str, output_dir: str = "chunks", debug: bool = False) -> List[Union[np.ndarray, dict]]:
    if file_path.lower().endswith('.pdf'):
        imgs = pdf_to_images(file_path)
        result = []
        for i,img in enumerate(imgs):
            page_dir = os.path.join(output_dir, f"page_{i+1}")
            result.extend(detect_and_split(img, page_dir, debug, True))
        return result
    elif file_path.lower().endswith(('.png','.jpg','.jpeg','.tiff')):
        return detect_and_split(file_path, output_dir, debug)
    elif file_path.lower().endswith(('.docx','.doc')):
        txt = extract_text_from_word(file_path)
        return split_into_questions(txt)
    else:
        raise ValueError(f"Unsupported type {file_path}")

if __name__ == '__main__':
    parts = process_document("test1.pdf", output_dir="chunks", debug=True)
    print(f"Generated {len(parts)} chunks/questions")

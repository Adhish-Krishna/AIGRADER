import cv2 as cv
import numpy as np
import os
from pdf2image import convert_from_path
from typing import List, Union

def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    try:
        pages = convert_from_path(pdf_path)
        images = []
        for page in pages:
            open_cv_image = cv.cvtColor(np.array(page), cv.COLOR_RGB2GRAY)
            images.append(open_cv_image)
            break
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def process_document(file_path: str, output_dir: str = "chunks", debug: bool = False) -> List[np.ndarray]:
    if file_path.lower().endswith('.pdf'):
        images = pdf_to_images(file_path)
        all_chunks = []
        for i, img in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            page_dir = os.path.join(output_dir, f"page_{i+1}")
            chunks = detect_and_split(img, output_dir=page_dir, debug=debug, is_array=True)
            all_chunks.extend(chunks)
        return all_chunks
    else:
        return detect_and_split(file_path, output_dir=output_dir, debug=debug)

def detect_and_split(image_input: Union[str, np.ndarray], output_dir="chunks", debug=False, is_array=False):
    if is_array:
        src = image_input
    else:
        src = cv.imread(image_input, cv.IMREAD_GRAYSCALE)
        if src is None:
            raise FileNotFoundError(f"Could not load {image_input}")
    
    h, w = src.shape

    edges = cv.Canny(src, 50, 150, apertureSize=3)

    linesP = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=int(h*.75),
        maxLineGap=10
    )

    y_hits = []
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:, 0]:
            if abs(y1 - y2) <= 5:
                y_hits.append((y1 + y2) * 0.5)

    if len(y_hits) == 0:
        print("----> No horizontal lines found — returning whole image as one chunk.")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "chunk_0.png")
        cv.imwrite(out_path, src)
        print(f"✔️ Saved chunk to {out_path}")
        return [src]

    y_hits = sorted(y_hits)
    merged = [y_hits[0]]
    tol = 10 
    for y in y_hits[1:]:
        if abs(y - merged[-1]) <= tol:
            merged[-1] = 0.5 * (merged[-1] + y)
        else:
            merged.append(y)

    cuts = [0] + merged + [h]

    if debug:
        vis = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        for y in merged:
            cv.line(vis, (0, int(y)), (w, int(y)), (0, 0, 255), 2)
        cv.imshow("Detected Cuts", cv.resize(vis, (480, 640)))
        cv.waitKey(0)

    os.makedirs(output_dir, exist_ok=True) 
    chunks = []
    for i in range(len(cuts) - 1):
        y0 = int(cuts[i] + tol / 2)
        y1 = int(cuts[i + 1] - tol / 2)
        if y1 <= y0:
            continue
        band = src[y0:y1, :]
        chunks.append(band)
        out_path = os.path.join(output_dir, f"chunk_{i}.png")
        try:
            cv.imwrite(out_path, band)
            print(f"Saved chunk {i} to {out_path}")
        except Exception as e:
            print(f"Failed to save chunk {i}: {e}")

        if debug:
            cv.imshow(f"Chunk {i}", cv.resize(band, (480, 200)))
            cv.waitKey(0)

    if debug:
        cv.destroyAllWindows()

    print(f"✔️ Total chunks saved: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    parts = process_document(
        "test.pdf", 
        output_dir="chunks",
        debug=True
    )
